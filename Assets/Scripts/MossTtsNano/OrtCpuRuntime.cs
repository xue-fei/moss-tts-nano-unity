using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// ONNX 运行时基类 - 对应 Python 的 ort_cpu_runtime.py OrtCpuRuntime
    /// 负责模型加载、推理会话管理、音频帧生成
    /// </summary>
    public class OrtCpuRuntime : IDisposable
    {
        protected const string SampleModeGreedy = "greedy";
        protected const string SampleModeFixed = "fixed";
        protected const string SampleModeFull = "full";
        protected const string ExecutionProviderCpu = "cpu";
        protected const string ExecutionProviderCuda = "cuda";

        protected ModelManifest _manifest;
        protected TtsModelMeta _ttsMeta;
        protected CodecModelMeta _codecMeta;
        protected string _modelDir;
        protected string _ttsDir;
        protected string _codecDir;
        protected int _threadCount;
        protected string _executionProvider;
        protected int _nVq;
        protected int _codebookSize;
        protected System.Random _rng;
        protected bool _disposed;

        protected IInferenceEngine _engine;

        public ModelManifest Manifest => _manifest;
        public TtsModelMeta TtsMeta => _ttsMeta;
        public CodecModelMeta CodecMeta => _codecMeta;

        public OrtCpuRuntime(
            string modelDir,
            int threadCount = 4,
            int? maxNewFrames = null,
            bool? doSample = null,
            string sampleMode = null,
            string executionProvider = ExecutionProviderCpu)
        {
            _modelDir = Path.GetFullPath(modelDir);
            _threadCount = Math.Max(1, threadCount);
            _executionProvider = NormalizeExecutionProvider(executionProvider);
            _rng = new System.Random(1234);

            LoadManifest();
            LoadTtsMeta();
            LoadCodecMeta();

            if (maxNewFrames.HasValue)
                _manifest.generation_defaults.max_new_frames = maxNewFrames.Value;
            if (doSample.HasValue)
                _manifest.generation_defaults.do_sample = doSample.Value;

            _manifest.generation_defaults.sample_mode = NormalizeSampleMode(
                sampleMode ?? _manifest.generation_defaults.sample_mode,
                _manifest.generation_defaults.do_sample);
            _manifest.generation_defaults.do_sample = _manifest.generation_defaults.sample_mode != SampleModeGreedy;

            _nVq = _manifest.tts_config.n_vq;
            _codebookSize = _ttsMeta.model_config.audio_codebook_sizes[0];

            // 创建推理引擎
            _engine = InferenceEngineFactory.Create("onnxruntime");
            _engine.LoadManifest(Path.Combine(_modelDir, "browser_poc_manifest.json"));
            _engine.InitializeSessions(threadCount);
        }

        private void LoadManifest()
        {
            // modelDir 由调用方解析为 StreamingAssets/Models/MOSS-TTS-Nano-ONNX，
            // manifest 就在该目录根下。
            string manifestPath = Path.Combine(_modelDir, "browser_poc_manifest.json");

            if (!File.Exists(manifestPath))
                throw new FileNotFoundException(
                    $"browser_poc_manifest.json not found at {manifestPath}. " +
                    $"模型应位于 StreamingAssets/Models/MOSS-TTS-Nano-ONNX 下。");

            string json = File.ReadAllText(manifestPath);
            _manifest = JsonConvert.DeserializeObject<ModelManifest>(json);
            Debug.Log($"[OrtCpuRuntime] Manifest loaded from {manifestPath}");
            Debug.Log($"[OrtCpuRuntime] prompt_templates: {(_manifest.prompt_templates != null ? "OK" : "NULL")}");
            Debug.Log($"[OrtCpuRuntime] tts_config: {(_manifest.tts_config != null ? "OK" : "NULL")}");
            Debug.Log($"[OrtCpuRuntime] builtin_voices: {(_manifest.builtin_voices?.Length.ToString() ?? "NULL")}");
        }

        private void LoadTtsMeta()
        {
            // tts_meta 在 manifest 同目录（manifest 里就是裸文件名）。
            string path = Path.Combine(_modelDir, _manifest.model_files.tts_meta);

            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"tts meta not found at {path} " +
                    $"(manifest declared '{_manifest.model_files.tts_meta}')。");

            string json = File.ReadAllText(path);
            _ttsMeta = JsonConvert.DeserializeObject<TtsModelMeta>(json);
            _ttsDir = Path.GetDirectoryName(path);
            Debug.Log($"[OrtCpuRuntime] TtsMeta loaded: prefill={_ttsMeta.files.prefill}, has_onnx={_ttsMeta.onnx != null}");
        }

        private void LoadCodecMeta()
        {
            // codec 目录嵌套在模型目录内，而 manifest 里写的是同级布局的 "../" 路径，
            // 两种布局的兼容逻辑统一放在 ModelPaths 里（OnnxRuntimeEngine 共用同一实现）。
            string path = ModelPaths.ResolveCodecMeta(_modelDir, _manifest.model_files.codec_meta);

            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"codec meta not found at {path} " +
                    $"(manifest declared '{_manifest.model_files.codec_meta}')。" +
                    $"请确认 StreamingAssets/Models/MOSS-TTS-Nano-ONNX 下的 codec 子目录完整。");

            string json = File.ReadAllText(path);
            _codecMeta = JsonConvert.DeserializeObject<CodecModelMeta>(json);
            _codecDir = Path.GetDirectoryName(path);
            Debug.Log($"[OrtCpuRuntime] CodecMeta loaded: sample_rate={_codecMeta.codec_config.sample_rate}, streaming_decode={_codecMeta.streaming_decode != null}");
        }

        /// <summary>
        /// 构建文本 token 行
        /// </summary>
        public List<int[]> BuildTextRows(List<int> tokenIds)
        {
            var rows = new List<int[]>();
            int rowWidth = _nVq + 1;
            foreach (int tokenId in tokenIds)
            {
                int[] row = new int[rowWidth];
                Array.Fill(row, _manifest.tts_config.audio_pad_token_id);
                row[0] = tokenId;
                rows.Add(row);
            }
            return rows;
        }

        /// <summary>
        /// 构建音频前缀行
        /// </summary>
        public List<int[]> BuildAudioPrefixRows(List<int[]> promptAudioCodes, int? slotTokenId = null)
        {
            var rows = new List<int[]>();
            int rowWidth = _nVq + 1;
            int resolvedSlotTokenId = slotTokenId ?? _manifest.tts_config.audio_user_slot_token_id;

            foreach (int[] codeRow in promptAudioCodes)
            {
                int[] row = new int[rowWidth];
                Array.Fill(row, _manifest.tts_config.audio_pad_token_id);
                row[0] = resolvedSlotTokenId;
                for (int i = 0; i < Math.Min(codeRow.Length, _nVq); i++)
                    row[i + 1] = codeRow[i];
                rows.Add(row);
            }
            return rows;
        }

        /// <summary>
        /// 构建语音克隆请求行
        /// </summary>
        public (List<int[]> inputIds, List<int[]> attentionMask) BuildVoiceCloneRequestRows(
            List<int[]> promptAudioCodes, List<int> textTokenIds)
        {
            if (_manifest.prompt_templates == null)
                throw new InvalidOperationException("prompt_templates is null in manifest");
            if (_manifest.prompt_templates.user_prompt_prefix_token_ids == null)
                throw new InvalidOperationException("user_prompt_prefix_token_ids is null");
            if (_manifest.prompt_templates.user_prompt_after_reference_token_ids == null)
                throw new InvalidOperationException("user_prompt_after_reference_token_ids is null");
            if (_manifest.prompt_templates.assistant_prompt_prefix_token_ids == null)
                throw new InvalidOperationException("assistant_prompt_prefix_token_ids is null");
            if (_manifest.tts_config == null)
                throw new InvalidOperationException("tts_config is null");

            var prefixTokenIds = new List<int>(_manifest.prompt_templates.user_prompt_prefix_token_ids);
            prefixTokenIds.Add(_manifest.tts_config.audio_start_token_id);

            var suffixTokenIds = new List<int> { _manifest.tts_config.audio_end_token_id };
            suffixTokenIds.AddRange(_manifest.prompt_templates.user_prompt_after_reference_token_ids);
            suffixTokenIds.AddRange(textTokenIds);
            suffixTokenIds.AddRange(_manifest.prompt_templates.assistant_prompt_prefix_token_ids);
            suffixTokenIds.Add(_manifest.tts_config.audio_start_token_id);

            var rows = new List<int[]>();
            rows.AddRange(BuildTextRows(prefixTokenIds));
            rows.AddRange(BuildAudioPrefixRows(promptAudioCodes));
            rows.AddRange(BuildTextRows(suffixTokenIds));

            // Attention mask: [[1, 1, 1, ..., 1]] with length = number of rows
            var attentionMask = new List<int[]> { Enumerable.Repeat(1, rows.Count).ToArray() };
            return (rows, attentionMask);
        }

        /// <summary>
        /// 生成音频帧 - 核心推理循环
        /// </summary>
        public List<int[]> GenerateAudioFrames(
            (List<int[]> inputIds, List<int[]> attentionMask) requestRows,
            Action<List<int[]>, int, int[]> onFrame = null)
        {
            var config = _manifest.generation_defaults;
            int rowWidth = _nVq + 1;

            // Prefill
            var (globalHidden, pastStates, pastValidLength) = Prefill(requestRows.inputIds, requestRows.attentionMask);

            var generatedFrames = new List<int[]>();
            var previousTokensByChannel = new List<List<int>>();
            var previousTokenSetsByChannel = new List<HashSet<int>>();
            for (int i = 0; i < _nVq; i++)
            {
                previousTokensByChannel.Add(new List<int>());
                previousTokenSetsByChannel.Add(new HashSet<int>());
            }

            for (int step = 0; step < config.max_new_frames; step++)
            {
                var frame = new List<int>();
                bool shouldContinue;
                int[] frameTokens;

                if (_engine.HasLocalGreedyFrame && !config.do_sample)
                {
                    (shouldContinue, frameTokens) = _engine.LocalGreedyFrame(
                        globalHidden, previousTokenSetsByChannel, config.audio_repetition_penalty);
                    if (!shouldContinue) break;
                    frame = frameTokens.ToList();
                    TrackFrameTokens(frame, previousTokensByChannel, previousTokenSetsByChannel);
                }
                else if (_engine.HasLocalFixedSampledFrame && config.sample_mode == SampleModeFixed)
                {
                    (shouldContinue, frameTokens) = _engine.LocalFixedSampledFrame(
                        globalHidden, previousTokenSetsByChannel, _rng);
                    if (!shouldContinue) break;
                    frame = frameTokens.ToList();
                    TrackFrameTokens(frame, previousTokensByChannel, previousTokenSetsByChannel);
                }
                else if (_engine.HasLocalCachedStep)
                {
                    // 本地缓存步 - 逐通道采样。
                    //
                    // localPast 必须在每次调用后被返回值覆盖：local_cached_step 是
                    // 带 KV cache 的自回归会话，第 k 次调用要看到前 k-1 步的 key/value。
                    // 原实现把第 2 次调用的 nextPast2 固定喂给后续 15 个通道，
                    // 而 past_valid_lengths 仍逐步 +1，等于告诉模型"缓存里有 j 个
                    // 有效位置"但实际只给了 2 个，注意力读到的是未初始化区域。
                    // 对应 Python 的 local_past_by_name = ... 每步重新赋值。
                    var localPast = CreateEmptyLocalCachedPast();
                    int localPastValidLength = 0;

                    var (textLogits, _, pastAfterProbe) = _engine.LocalCachedStep(
                        globalHidden, 0, 0, 0, 0, localPastValidLength, localPast);
                    localPast = pastAfterProbe;
                    localPastValidLength++;

                    int nextTextToken = MossTtsMath.SampleAssistantTextToken(
                        textLogits,
                        _manifest.tts_config.audio_assistant_slot_token_id,
                        _manifest.tts_config.audio_end_token_id,
                        config,
                        _rng);

                    if (nextTextToken != _manifest.tts_config.audio_assistant_slot_token_id)
                        break;

                    var (_, audioLogits, pastAfterSlot) = _engine.LocalCachedStep(
                        globalHidden, nextTextToken, 0, 0, 1, localPastValidLength, localPast);
                    localPast = pastAfterSlot;
                    localPastValidLength++;

                    float[] firstChannelLogits = _engine.SliceAudioChannelLogits(audioLogits, 0, _nVq, _codebookSize);
                    int sampledToken = MossTtsMath.SampleAudioToken(
                        firstChannelLogits,
                        previousTokensByChannel[0],
                        previousTokenSetsByChannel[0],
                        config,
                        _rng);
                    frame.Add(sampledToken);
                    previousTokensByChannel[0].Add(sampledToken);
                    previousTokenSetsByChannel[0].Add(sampledToken);

                    int previousToken = sampledToken;
                    for (int ch = 1; ch < _nVq; ch++)
                    {
                        var (_, aLogits, pastAfterChannel) = _engine.LocalCachedStep(
                            globalHidden, 0, previousToken, ch - 1, 2, localPastValidLength, localPast);
                        localPast = pastAfterChannel;
                        localPastValidLength++;

                        float[] channelLogits = _engine.SliceAudioChannelLogits(aLogits, ch, _nVq, _codebookSize);
                        sampledToken = MossTtsMath.SampleAudioToken(
                            channelLogits,
                            previousTokensByChannel[ch],
                            previousTokenSetsByChannel[ch],
                            config,
                            _rng);
                        frame.Add(sampledToken);
                        previousTokensByChannel[ch].Add(sampledToken);
                        previousTokenSetsByChannel[ch].Add(sampledToken);
                        previousToken = sampledToken;
                    }
                }
                else
                {
                    // 本地解码器回退
                    var (textLogits, _) = _engine.LocalDecoder(globalHidden, 0, Array.Empty<int>());
                    int nextTextToken = MossTtsMath.SampleAssistantTextToken(
                        textLogits,
                        _manifest.tts_config.audio_assistant_slot_token_id,
                        _manifest.tts_config.audio_end_token_id,
                        config,
                        _rng);

                    if (nextTextToken != _manifest.tts_config.audio_assistant_slot_token_id)
                        break;

                    for (int ch = 0; ch < _nVq; ch++)
                    {
                        var (_, audioLogits) = _engine.LocalDecoder(globalHidden, nextTextToken, frame.ToArray());
                        float[] channelLogits = _engine.SliceAudioChannelLogits(audioLogits, ch, _nVq, _codebookSize);
                        int sampledToken = MossTtsMath.SampleAudioToken(
                            channelLogits,
                            previousTokensByChannel[ch],
                            previousTokenSetsByChannel[ch],
                            config,
                            _rng);
                        frame.Add(sampledToken);
                        previousTokensByChannel[ch].Add(sampledToken);
                        previousTokenSetsByChannel[ch].Add(sampledToken);
                    }
                }

                generatedFrames.Add(frame.ToArray());
                onFrame?.Invoke(generatedFrames, step, frame.ToArray());

                // Decode step - 更新 global hidden 和 past states
                int[] nextRow = new int[rowWidth];
                Array.Fill(nextRow, _manifest.tts_config.audio_pad_token_id);
                nextRow[0] = _manifest.tts_config.audio_assistant_slot_token_id;
                for (int i = 0; i < frame.Count; i++)
                    nextRow[i + 1] = frame[i];

                var (newGlobalHidden, newPastStates) = DecodeSingleStep(nextRow, pastValidLength, pastStates);
                globalHidden = newGlobalHidden;
                pastStates = newPastStates;
                pastValidLength++;
            }

            return generatedFrames;
        }

        /// <summary>
        /// 记录整帧采样结果，供后续步骤的重复惩罚使用。
        /// 走 LocalGreedyFrame / LocalFixedSampledFrame 这类"一次出整帧"的路径时，
        /// 采样发生在 ONNX 图内部，必须在 C# 侧手动回填历史，否则 repetition_seen_mask 永远为空。
        /// </summary>
        private static void TrackFrameTokens(
            List<int> frame,
            List<List<int>> previousTokensByChannel,
            List<HashSet<int>> previousTokenSetsByChannel)
        {
            int channels = Math.Min(frame.Count, previousTokensByChannel.Count);
            for (int ch = 0; ch < channels; ch++)
            {
                previousTokensByChannel[ch].Add(frame[ch]);
                previousTokenSetsByChannel[ch].Add(frame[ch]);
            }
        }

        /// <summary>
        /// 单步解码
        /// </summary>
        protected (float[] globalHidden, Dictionary<string, float[]> newPastStates) DecodeSingleStep(
            int[] inputIds, int pastValidLength, Dictionary<string, float[]> pastStates)
        {
            var inputs = new int[1, 1, inputIds.Length];
            for (int j = 0; j < inputIds.Length; j++)
                inputs[0, 0, j] = inputIds[j];

            return _engine.Decode(inputs, pastValidLength, pastStates);
        }

        /// <summary>
        /// 创建空的本地缓存 past
        /// </summary>
        protected Dictionary<string, float[]> CreateEmptyLocalCachedPast()
        {
            var past = new Dictionary<string, float[]>();
            int layers = _ttsMeta.model_config.local_layers;
            int heads = _ttsMeta.model_config.local_heads;
            int headDim = _ttsMeta.model_config.local_head_dim;

            for (int l = 0; l < layers; l++)
            {
                past[$"local_past_key_{l}"] = new float[0];
                past[$"local_past_value_{l}"] = new float[0];
            }

            return past;
        }

        protected virtual (float[] globalHidden, Dictionary<string, float[]> pastStates, int pastValidLength) Prefill(
            List<int[]> inputIds, List<int[]> attentionMask)
        {
            var inputs = new int[1, inputIds.Count, inputIds[0].Length];
            for (int i = 0; i < inputIds.Count; i++)
                for (int j = 0; j < inputIds[i].Length; j++)
                    inputs[0, i, j] = inputIds[i][j];

            // attentionMask is [[1,1,...,1]] with length = inputIds.Count
            var masks = new int[1, attentionMask[0].Length];
            for (int i = 0; i < attentionMask[0].Length; i++)
                masks[0, i] = attentionMask[0][i];

            return _engine.Prefill(inputs, masks);
        }

        /// <summary>
        /// 归一化采样模式
        /// </summary>
        protected static string NormalizeSampleMode(string sampleMode, bool doSample)
        {
            string mode = sampleMode?.Trim().ToLower();
            if (mode == SampleModeGreedy || mode == SampleModeFixed || mode == SampleModeFull)
                return mode;
            return doSample ? SampleModeFixed : SampleModeGreedy;
        }

        /// <summary>
        /// 归一化执行提供者
        /// </summary>
        protected static string NormalizeExecutionProvider(string provider)
        {
            string p = provider?.Trim().ToLower();
            if (p == "cpu" || p == "cpuexecutionprovider")
                return ExecutionProviderCpu;
            if (p == "cuda" || p == "gpu" || p == "cudaexecutionprovider")
                return ExecutionProviderCuda;
            throw new ArgumentException("execution_provider must be one of: cpu, cuda");
        }

        /// <summary>
        /// 计算流式解码帧预算
        /// </summary>
        protected int ResolveStreamDecodeFrameBudget(int emittedSamplesTotal, int sampleRate, float? firstAudioEmittedAt)
        {
            if (!firstAudioEmittedAt.HasValue) return 1;

            float elapsed = Time.realtimeSinceStartup - firstAudioEmittedAt.Value;
            float emittedSeconds = emittedSamplesTotal / (float)sampleRate;
            float leadSeconds = emittedSeconds - elapsed;

            if (leadSeconds < 0.20f) return 1;
            if (leadSeconds < 0.55f) return 2;
            if (leadSeconds < 1.10f) return 4;
            return 8;
        }

        public virtual void Dispose()
        {
            if (!_disposed)
            {
                _engine?.Dispose();
                _disposed = true;
            }
        }
    }
}
