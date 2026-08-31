using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// ONNX TTS 运行时 - 对应 Python 的 onnx_tts_runtime.py OnnxTtsRuntime
    /// 封装文本编码、语音克隆分块、音频合成等高层逻辑
    /// </summary>
    public class OnnxTtsRuntime : OrtCpuRuntime
    {
        private readonly string _outputDir;
        private readonly Dictionary<string, int> _tokenCache;
        private readonly Dictionary<string, List<int[]>> _voiceCache;
        private ITokenizer _tokenizer;

        /// <summary>
        /// 是否压缩退化静音段（<see cref="SilentFrameGuard"/>）。
        ///
        /// 原版 Python 没有这一步。它是分词器出错时期加的补偿：当时全角标点被拆成
        /// byte-fallback token，模型收到偏离训练分布的输入后会连续几十帧输出静音
        /// token。分词器与官方 sentencepiece 对齐后根因已消除，而丢帧本身会压缩
        /// 时间轴、削掉正常句读停顿，所以默认关闭；仅在排查退化时临时打开。
        /// </summary>
        public bool CollapseDegenerateSilence { get; set; } = false;

        /// <summary>
        /// 生成完成后回调整帧序列，供离线校验工具导出 token 做数值对比。
        /// 生产路径不设置此委托，因此没有开销。
        /// </summary>
        public Action<List<int[]>> FrameTraceSink { get; set; }

        public OnnxTtsRuntime(
            string modelDir,
            int threadCount = 4,
            int? maxNewFrames = null,
            bool? doSample = null,
            string sampleMode = null,
            string executionProvider = "cuda",
            string outputDir = null)
            : base(modelDir, threadCount, maxNewFrames, doSample, sampleMode, executionProvider)
        {
            // 原来这里无条件用 Application.dataPath，既忽略了传入的 outputDir，
            // 又会把 WAV 写进 Assets/（Editor 下污染工程，打包后 dataPath 不可写）。
            _outputDir = string.IsNullOrEmpty(outputDir)
                ? Path.Combine(Application.persistentDataPath, "MossTtsOutput")
                : outputDir;
            Directory.CreateDirectory(_outputDir);
            _tokenCache = new Dictionary<string, int>();
            _voiceCache = new Dictionary<string, List<int[]>>();

            // 加载 SentencePiece 分词器
            InitializeTokenizer(modelDir);
        }

        private void InitializeTokenizer(string modelDir)
        {
            // tokenizer_sp.json + tokenizer_charsmap.bytes 与模型同目录，
            // 由 MOSS-TTS-Nano/export_tokenizer_assets.py 从 tokenizer.model 导出。
            // 旧的 tokenizer_vocab_parallel.json 只有 pieces/scores，缺 types 与
            // nmt_nfkc 归一化规则表，无法复现官方切分，已不再使用。
            string spJsonPath = Path.GetFullPath(Path.Combine(_modelDir, "tokenizer_sp.json"));

            if (File.Exists(spJsonPath))
            {
                try
                {
                    _tokenizer = new SentencePieceTokenizer(spJsonPath);
                    Debug.Log($"[OnnxTtsRuntime] Tokenizer loaded from {spJsonPath}");
                    return;
                }
                catch (Exception e)
                {
                    Debug.LogError($"[OnnxTtsRuntime] Failed to load tokenizer from {spJsonPath}: {e}");
                }
            }
            else
            {
                Debug.LogError(
                    $"[OnnxTtsRuntime] {spJsonPath} not found. Generate it with:\n" +
                    "  python MOSS-TTS-Nano/export_tokenizer_assets.py " +
                    "<model_dir>/tokenizer.model <model_dir>");
            }

            // 回退分词器的 id 与模型词表毫无关系，合成结果必然是噪声，
            // 这里明确告知而不是让它静默产出坏音频。
            Debug.LogError(
                "[OnnxTtsRuntime] Falling back to SimpleUnicodeTokenizer — " +
                "synthesized audio WILL be wrong until the SentencePiece assets are present.");
            _tokenizer = new SimpleUnicodeTokenizer();
        }

        /// <summary>
        /// 编码文本为 token IDs
        /// </summary>
        public List<int> EncodeText(string text)
        {
            if (string.IsNullOrEmpty(text)) return new List<int>();
            var tokens = _tokenizer.Encode(text);
            if (tokens.Count > 0)
            {
                string preview = string.Join(", ", tokens.Take(10));
                if (tokens.Count > 10) preview += "...";
                Debug.Log($"[OnnxTtsRuntime] EncodeText '{text}' -> [{preview}] ({tokens.Count} tokens)");
            }
            return tokens;
        }

        /// <summary>
        /// 计算文本 token 数量
        /// </summary>
        public int CountTextTokens(string text)
        {
            return _tokenizer.CountTokens(text);
        }

        /// <summary>
        /// 准备合成文本（归一化）
        /// </summary>
        public (string text, string normalizationMethod, string language) PrepareSynthesisText(
            string text, string voice = "", bool enableWeText = true, bool enableNormalize = true)
        {
            string normalized = text ?? "";
            string method = "none";
            string language = "zh";

            if (enableNormalize)
            {
                normalized = NormalizeTtsText(normalized);
                method = "normalize";
            }

            if (enableWeText)
            {
                // WeTextProcessing 归一化（需要额外库支持）
                // 这里仅做简单处理
                language = TextChunker.ContainsCjk(normalized) ? "zh" : "en";
                method += "+wetext";
            }

            return (normalized, method, language);
        }

        /// <summary>
        /// 简单 TTS 文本归一化
        /// </summary>
        private string NormalizeTtsText(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            // 基础清理
            text = text.Trim();
            text = System.Text.RegularExpressions.Regex.Replace(text, @"\s+", " ");

            // 移除控制字符
            var sb = new StringBuilder();
            foreach (char c in text)
            {
                if (!char.IsControl(c) || c == '\n' || c == '\t')
                    sb.Append(c);
            }

            return sb.ToString();
        }

        /// <summary>
        /// 按 token 预算切分文本
        /// </summary>
        public List<string> SplitTextByTokenBudget(string text, int maxTokens)
        {
            return TextChunker.SplitByTokenBudget(text, maxTokens, CountTextTokens);
        }

        /// <summary>
        /// 语音克隆文本分块
        /// </summary>
        public List<string> SplitVoiceCloneText(string text, int maxTokens = 75)
        {
            return TextChunker.SplitVoiceCloneText(text, maxTokens, CountTextTokens);
        }

        /// <summary>
        /// 估算块间停顿
        /// </summary>
        public float EstimateInterChunkPauseSeconds(string textChunk)
        {
            return TextChunker.EstimateInterChunkPauseSeconds(textChunk);
        }

        /// <summary>
        /// 编码参考音频
        /// </summary>
        public List<int[]> EncodeReferenceAudio(string audioPath)
        {
            if (_voiceCache.TryGetValue(audioPath, out var cached))
                return cached;

            // 加载音频文件
            float[] waveform = LoadAudio(audioPath);
            int sampleRate = _codecMeta.codec_config.sample_rate;

            // 重采样如果需要
            // TODO: 实现重采样

            // 使用 codec encoder 编码
            var codes = _engine.CodecEncode(waveform, sampleRate);
            _voiceCache[audioPath] = codes;
            return codes;
        }

        /// <summary>
        /// 获取内置语音的 prompt audio codes
        /// </summary>
        public List<int[]> GetBuiltinVoiceCodes(string voiceName)
        {
            var voice = _manifest.builtin_voices?.FirstOrDefault(v => v.voice == voiceName);
            if (voice == null)
                throw new ArgumentException($"Built-in voice not found: {voiceName}");

            return voice.prompt_audio_codes?.ToList() ?? new List<int[]>();
        }

        /// <summary>
        /// 合成单个文本块
        /// </summary>
        public (float[] waveform, List<int[]> generatedFrames) SynthesizeSingleChunk(
            string text,
            List<int[]> promptAudioCodes,
            bool streaming = false)
        {
            List<int> textTokenIds = EncodeText(text);
            var (inputIds, attentionMask) = BuildVoiceCloneRequestRows(promptAudioCodes, textTokenIds);

            float[] waveform;
            List<int[]> generatedFrames;

            if (!streaming)
            {
                generatedFrames = GenerateAudioFrames((inputIds, attentionMask));
                FrameTraceSink?.Invoke(generatedFrames);
                if (generatedFrames.Count == 0)
                {
                    Debug.LogWarning($"[OnnxTtsRuntime] No audio frames generated for '{text}' (model stopped at step 0)");
                    return (Array.Empty<float>(), generatedFrames);
                }

                // 静音段压缩是 C# 独有的补丁，原版 Python 没有对应逻辑。
                // 它当初是为了掩盖分词错误导致的大段空白；分词修正后不再需要，
                // 而且丢帧会改变时间轴、削掉正常句读停顿，所以默认关闭。
                List<int[]> decodeFrames = generatedFrames;
                if (CollapseDegenerateSilence)
                {
                    decodeFrames = SilentFrameGuard.Collapse(generatedFrames);
                    if (decodeFrames.Count != generatedFrames.Count)
                    {
                        Debug.Log($"[OnnxTtsRuntime] Collapsed degenerate silence: " +
                                  $"{generatedFrames.Count} -> {decodeFrames.Count} frames");
                    }
                }

                var (channelArrays, audioLength) = _engine.CodecDecode(decodeFrames);
                waveform = MergeAudioChannels(channelArrays, audioLength);
                Debug.Log($"[OnnxTtsRuntime] Chunk done: {decodeFrames.Count} frames -> {audioLength} samples/ch");
                return (waveform, generatedFrames);
            }

            // 流式合成，对齐 Python synthesize_single_chunk 的 streaming 分支
            var emittedChunks = new List<float[]>();
            int emittedSamplesTotal = 0;
            float? firstAudioEmittedAt = null;
            var pendingDecodeFrames = new List<int[]>();

            // 每个 chunk 独立的流式解码状态，避免跨 chunk 缓存串味
            _engine.CodecDecodeStepReset();

            // 对应 Python 的 decode_pending_frames(force)
            void DecodePendingFrames(bool force)
            {
                int pendingCount = pendingDecodeFrames.Count;
                if (pendingCount <= 0) return;

                int decodeBudget = Math.Max(1, ResolveStreamDecodeFrameBudget(
                    emittedSamplesTotal, _codecMeta.codec_config.sample_rate, firstAudioEmittedAt));

                if (!force && pendingCount < decodeBudget) return;

                // 只取 budget 内的帧，剩下的留在队列里下次再解。
                // 原来用 Clear() 会把超出 budget 的帧也一并交给有状态的
                // decode_step，等价于跳过了它们的时序位置。
                int frameBudget = force ? pendingCount : Math.Min(pendingCount, decodeBudget);
                var frameChunk = pendingDecodeFrames.GetRange(0, frameBudget);
                pendingDecodeFrames.RemoveRange(0, frameBudget);

                var (audio, audioLength) = _engine.CodecDecodeStep(frameChunk);
                if (audioLength <= 0) return;

                if (!firstAudioEmittedAt.HasValue)
                    firstAudioEmittedAt = Time.realtimeSinceStartup;

                emittedSamplesTotal += audioLength;
                emittedChunks.Add(audio);
            }

            void OnFrame(List<int[]> frames, int step, int[] frame)
            {
                // Python 是 pending_decode_frames.append(list(frame))，即存副本。
                // 直接存引用的话，上游若复用同一数组，队列里的帧会被后续步骤改写。
                pendingDecodeFrames.Add((int[])frame.Clone());
                DecodePendingFrames(false);
            }

            try
            {
                generatedFrames = GenerateAudioFrames((inputIds, attentionMask), OnFrame);
                DecodePendingFrames(true);
            }
            finally
            {
                _engine.CodecDecodeStepReset();
            }

            waveform = ConcatWaveforms(emittedChunks);
            return (waveform, generatedFrames);
        }

        /// <summary>
        /// 完整合成流程
        /// </summary>
        public SynthesisResult Synthesize(
            string text,
            string voice = null,
            string promptAudioPath = null,
            string outputPath = null,
            string sampleMode = null,
            bool doSample = true,
            bool streaming = false,
            int? maxNewFrames = null,
            int voiceCloneMaxTextTokens = 75,
            bool enableWeText = true,
            bool enableNormalize = true,
            int? seed = null)
        {
            if (maxNewFrames.HasValue)
                _manifest.generation_defaults.max_new_frames = maxNewFrames.Value;

            string normalizedSampleMode = NormalizeSampleMode(sampleMode ?? _manifest.generation_defaults.sample_mode, doSample);
            _manifest.generation_defaults.sample_mode = normalizedSampleMode;
            _manifest.generation_defaults.do_sample = normalizedSampleMode != SampleModeGreedy;

            if (seed.HasValue)
                _rng = new System.Random(seed.Value);

            // 准备文本
            var (preparedText, method, language) = PrepareSynthesisText(text, voice ?? "", enableWeText, enableNormalize);

            // 获取 prompt audio codes
            List<int[]> promptAudioCodes;
            if (!string.IsNullOrEmpty(promptAudioPath))
            {
                promptAudioCodes = EncodeReferenceAudio(promptAudioPath);
            }
            else
            {
                string resolvedVoice = voice ?? _manifest.builtin_voices?.FirstOrDefault()?.voice ?? "Junhao";
                promptAudioCodes = GetBuiltinVoiceCodes(resolvedVoice);
            }

            // 分块
            List<string> textChunks = SplitVoiceCloneText(preparedText, voiceCloneMaxTextTokens);

            // 逐块合成
            var allWaveforms = new List<float[]>();
            var allGeneratedFrames = new List<int[]>();
            var chunkResults = new List<(string text, float[] waveform)>();

            int sampleRate = _codecMeta.codec_config.sample_rate;
            int channels = _codecMeta.codec_config.channels;

            for (int i = 0; i < textChunks.Count; i++)
            {
                var (waveform, frames) = SynthesizeSingleChunk(textChunks[i], promptAudioCodes, streaming);
                allWaveforms.Add(waveform);
                allGeneratedFrames.AddRange(frames);
                chunkResults.Add((textChunks[i], waveform));

                // 添加块间停顿
                if (i < textChunks.Count - 1)
                {
                    float pauseSeconds = EstimateInterChunkPauseSeconds(textChunks[i]);
                    int pauseSamples = Mathf.Max(0, Mathf.RoundToInt(sampleRate * pauseSeconds));
                    if (pauseSamples > 0)
                        allWaveforms.Add(new float[pauseSamples * channels]);
                }
            }

            // 合并波形
            float[] finalWaveform = ConcatWaveforms(allWaveforms);

            // 保存文件
            string audioPath = outputPath ?? Path.Combine(_outputDir, $"moss_tts_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
            WriteWaveformToWav(audioPath, finalWaveform, sampleRate, channels);

            return new SynthesisResult
            {
                AudioPath = audioPath,
                Waveform = finalWaveform,
                SampleRate = sampleRate,
                Channels = channels,
                Voice = voice ?? "Junhao",
                Mode = "voice_clone",
                TextChunks = textChunks.ToArray()
            };
        }

        /// <summary>
        /// 流式合成 - 返回音频块事件
        /// </summary>
        public List<AudioChunkEvent> SynthesizeStream(
            string text,
            string voice = null,
            string promptAudioPath = null,
            int? maxNewFrames = null,
            int voiceCloneMaxTextTokens = 75,
            bool enableWeText = true,
            bool enableNormalize = true,
            int? seed = null)
        {
            if (maxNewFrames.HasValue)
                _manifest.generation_defaults.max_new_frames = maxNewFrames.Value;

            if (seed.HasValue)
                _rng = new System.Random(seed.Value);

            var (preparedText, _, _) = PrepareSynthesisText(text, voice ?? "", enableWeText, enableNormalize);

            List<int[]> promptAudioCodes;
            if (!string.IsNullOrEmpty(promptAudioPath))
                promptAudioCodes = EncodeReferenceAudio(promptAudioPath);
            else
                promptAudioCodes = GetBuiltinVoiceCodes(voice ?? "Junhao");

            List<string> textChunks = SplitVoiceCloneText(preparedText, voiceCloneMaxTextTokens);
            int sampleRate = _codecMeta.codec_config.sample_rate;
            int channels = _codecMeta.codec_config.channels;
            int chunkIndex = 0;
            var events = new List<AudioChunkEvent>();

            foreach (string chunkText in textChunks)
            {
                int emittedSamplesTotal = 0;
                float? firstAudioEmittedAt = null;

                List<int> textTokenIds = EncodeText(chunkText);
                var (inputIds, attentionMask) = BuildVoiceCloneRequestRows(promptAudioCodes, textTokenIds);

                var pendingDecodeFrames = new List<int[]>();
                _engine.CodecDecodeStepReset();

                // 与 SynthesizeSingleChunk 一致的 pending 队列语义：
                // decode_step 是有状态会话，必须按 budget 分批消费、消费后移除。
                void DecodePendingFrames(bool force)
                {
                    int pendingCount = pendingDecodeFrames.Count;
                    if (pendingCount <= 0) return;

                    int decodeBudget = Math.Max(1, ResolveStreamDecodeFrameBudget(
                        emittedSamplesTotal, sampleRate, firstAudioEmittedAt));

                    if (!force && pendingCount < decodeBudget) return;

                    int frameBudget = force ? pendingCount : Math.Min(pendingCount, decodeBudget);
                    var frameChunk = pendingDecodeFrames.GetRange(0, frameBudget);
                    pendingDecodeFrames.RemoveRange(0, frameBudget);

                    var (audio, audioLength) = _engine.CodecDecodeStep(frameChunk);
                    if (audioLength <= 0) return;

                    if (!firstAudioEmittedAt.HasValue)
                        firstAudioEmittedAt = Time.realtimeSinceStartup;

                    emittedSamplesTotal += audioLength;

                    float emittedSeconds = emittedSamplesTotal / (float)sampleRate;
                    float leadSeconds = emittedSeconds - (Time.realtimeSinceStartup - firstAudioEmittedAt.Value);

                    events.Add(new AudioChunkEvent
                    {
                        Waveform = audio,
                        SampleRate = sampleRate,
                        ChunkIndex = chunkIndex,
                        IsPause = false,
                        EmittedAudioSeconds = emittedSeconds,
                        LeadSeconds = leadSeconds
                    });
                }

                void OnFrame(List<int[]> frames, int step, int[] frame)
                {
                    pendingDecodeFrames.Add((int[])frame.Clone());
                    DecodePendingFrames(false);
                }

                try
                {
                    GenerateAudioFrames((inputIds, attentionMask), OnFrame);
                    DecodePendingFrames(true);
                }
                finally
                {
                    _engine.CodecDecodeStepReset();
                }

                chunkIndex++;
            }

            return events;
        }

        #region Audio Utilities

        /// <summary>
        /// 加载音频文件（支持 WAV）
        /// </summary>
        private float[] LoadAudio(string path)
        {
            using var reader = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var waveReader = new BinaryReader(reader);

            // 读取 WAV 头
            waveReader.ReadBytes(4); // RIFF
            waveReader.ReadInt32(); // file size
            waveReader.ReadBytes(4); // WAVE
            waveReader.ReadBytes(4); // fmt
            waveReader.ReadInt32(); // fmt size
            waveReader.ReadInt16(); // format
            int channels = waveReader.ReadInt16();
            int sampleRate = waveReader.ReadInt32();
            waveReader.ReadInt32(); // byte rate
            waveReader.ReadInt16(); // block align
            int bitsPerSample = waveReader.ReadInt16();

            // 查找 data chunk
            while (reader.Position < reader.Length)
            {
                byte[] chunkId = waveReader.ReadBytes(4);
                int chunkSize = waveReader.ReadInt32();
                string chunkName = System.Text.Encoding.ASCII.GetString(chunkId);

                if (chunkName == "data")
                {
                    byte[] data = waveReader.ReadBytes(chunkSize);
                    return ConvertBytesToFloat(data, bitsPerSample, channels);
                }
                else
                {
                    waveReader.ReadBytes(chunkSize);
                }
            }

            return Array.Empty<float>();
        }

        private float[] ConvertBytesToFloat(byte[] data, int bitsPerSample, int channels)
        {
            int bytesPerSample = bitsPerSample / 8;
            int sampleCount = data.Length / bytesPerSample;
            float[] result = new float[sampleCount];

            for (int i = 0; i < sampleCount; i++)
            {
                int offset = i * bytesPerSample;
                float sample = 0f;

                if (bitsPerSample == 16)
                {
                    short s = BitConverter.ToInt16(data, offset);
                    sample = s / 32768f;
                }
                else if (bitsPerSample == 8)
                {
                    sample = (data[offset] - 128) / 128f;
                }
                else if (bitsPerSample == 24)
                {
                    int s = (data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16));
                    if ((s & 0x800000) != 0) s |= -0x1000000;
                    sample = s / 8388608f;
                }

                result[i] = Mathf.Clamp(sample, -1f, 1f);
            }

            return result;
        }

        /// <summary>
        /// 合并多通道音频
        /// </summary>
        private float[] MergeAudioChannels(float[][] channelArrays, int audioLength)
        {
            if (channelArrays.Length == 0) return Array.Empty<float>();
            if (channelArrays.Length == 1) return channelArrays[0];

            int channels = channelArrays.Length;
            float[] result = new float[audioLength * channels];

            for (int ch = 0; ch < channels; ch++)
            {
                for (int s = 0; s < audioLength; s++)
                {
                    result[s * channels + ch] = channelArrays[ch][s];
                }
            }

            return result;
        }

        /// <summary>
        /// 拼接多个波形
        /// </summary>
        private float[] ConcatWaveforms(List<float[]> waveforms)
        {
            int totalLength = 0;
            foreach (var w in waveforms)
                totalLength += w.Length;

            float[] result = new float[totalLength];
            int offset = 0;
            foreach (var w in waveforms)
            {
                Array.Copy(w, 0, result, offset, w.Length);
                offset += w.Length;
            }

            return result;
        }

        /// <summary>
        /// 写入 WAV 文件
        /// </summary>
        private void WriteWaveformToWav(string path, float[] waveform, int sampleRate, int channels)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));

            using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var writer = new BinaryWriter(stream);

            int bitsPerSample = 16;
            int byteRate = sampleRate * channels * bitsPerSample / 8;
            int blockAlign = channels * bitsPerSample / 8;
            int dataSize = waveform.Length * bitsPerSample / 8;

            // RIFF header
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(36 + dataSize);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

            // fmt chunk
            writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16);
            writer.Write((short)1); // PCM
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(byteRate);
            writer.Write((short)blockAlign);
            writer.Write((short)bitsPerSample);

            // data chunk
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(dataSize);

            foreach (float sample in waveform)
            {
                short s = (short)Mathf.RoundToInt(Mathf.Clamp(sample, -1f, 1f) * 32767f);
                writer.Write(s);
            }
        }

        #endregion
    }
}
