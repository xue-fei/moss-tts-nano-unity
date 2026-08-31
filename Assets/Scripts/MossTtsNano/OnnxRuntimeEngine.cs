using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// ONNX Runtime 推理引擎
    /// </summary>
    public class OnnxRuntimeEngine : IInferenceEngine
    {
        public bool HasLocalGreedyFrame => _localGreedyFrameSession != null;
        public bool HasLocalFixedSampledFrame => _localFixedSampledFrameSession != null;
        public bool HasLocalCachedStep => _localCachedStepSession != null;

        private InferenceSession _prefillSession;
        private InferenceSession _decodeSession;
        private InferenceSession _localDecoderSession;
        private InferenceSession _localGreedyFrameSession;
        private InferenceSession _localFixedSampledFrameSession;
        private InferenceSession _localCachedStepSession;
        private InferenceSession _codecEncodeSession;
        private InferenceSession _codecDecodeSession;
        private InferenceSession _codecDecodeStepSession;

        private ModelManifest _manifest;
        private TtsModelMeta _ttsMeta;
        private CodecModelMeta _codecMeta;
        private string _modelDir;
        private string _ttsDir;
        private string _codecDir;
        private int _nVq;
        private int _codebookSize;
        private int _threadCount;
        private bool _disposed;

        // KV cache 布局: [batch, seq, heads, headDim]
        private int _globalKvStride;   // global_heads * head_dim
        private int _localKvStride;    // local_heads * local_head_dim
        private int _globalHeads;
        private int _globalHeadDim;
        private int _localHeads;
        private int _localHeadDim;

        private Dictionary<string, NamedOnnxValue> _streamingInputs;

        // 复用缓冲区：repetition_seen_mask 每帧都是 [1, nVq, codebookSize] = 16×1024 个 int（64KB）。
        // 每帧新建会在 375 帧的生成过程中产生约 24MB 垃圾，Unity 下会触发可感知的 GC 卡顿。
        private int[] _repetitionMaskBuffer;
        private DenseTensor<int> _repetitionMaskTensor;
        private float[] _audioRandomBuffer;
        private DenseTensor<float> _audioRandomTensor;

        public void LoadManifest(string manifestPath)
        {
            string json = File.ReadAllText(manifestPath);
            _manifest = JsonConvert.DeserializeObject<ModelManifest>(json);
            _modelDir = Path.GetDirectoryName(manifestPath);
            _ttsDir = Path.Combine(_modelDir, Path.GetDirectoryName(_manifest.model_files.tts_meta));
            _nVq = _manifest.tts_config.n_vq;

            string ttsMetaJson = File.ReadAllText(Path.Combine(_modelDir, _manifest.model_files.tts_meta));
            _ttsMeta = JsonConvert.DeserializeObject<TtsModelMeta>(ttsMetaJson);
            _codebookSize = _ttsMeta.model_config.audio_codebook_sizes[0];

            // codec 目录嵌套在模型目录内，manifest 里却写的是同级布局的 "../" 路径，
            // 两种布局的兼容逻辑统一在 ModelPaths（OrtCpuRuntime 共用同一实现）。
            string codecMetaPath = ModelPaths.ResolveCodecMeta(_modelDir, _manifest.model_files.codec_meta);
            if (!File.Exists(codecMetaPath))
                throw new FileNotFoundException(
                    $"codec meta not found at {codecMetaPath} " +
                    $"(manifest declared '{_manifest.model_files.codec_meta}')。" +
                    $"请确认 StreamingAssets/Models/MOSS-TTS-Nano-ONNX 下的 codec 子目录完整。");

            string codecMetaJson = File.ReadAllText(codecMetaPath);
            _codecMeta = JsonConvert.DeserializeObject<CodecModelMeta>(codecMetaJson);
            _codecDir = Path.GetDirectoryName(codecMetaPath);

            Debug.Log($"[OnnxRuntime] Manifest loaded: {_manifest.builtin_voices?.Length} voices, nVq={_nVq}, codebook={_codebookSize}");
            Debug.Log($"[OnnxRuntime] prompt_templates: {(_manifest.prompt_templates != null ? "OK" : "NULL")}");

            // KV cache 形状信息: [batch, seq, heads, headDim]
            var mc = _ttsMeta.model_config;
            _globalHeads = mc.global_heads;
            _globalHeadDim = mc.head_dim;
            _localHeads = mc.local_heads;
            _localHeadDim = mc.local_head_dim;
            _globalKvStride = Math.Max(1, _globalHeads * _globalHeadDim);
            _localKvStride = Math.Max(1, _localHeads * _localHeadDim);
        }

        public void InitializeSessions(int threadCount = 4)
        {
            _threadCount = threadCount;
            var sessionOptions = CreateSessionOptions(threadCount);

            _prefillSession = LoadSession(sessionOptions, _ttsDir, _ttsMeta.files.prefill);
            _decodeSession = LoadSession(sessionOptions, _ttsDir, _ttsMeta.files.decode_step);
            _localDecoderSession = LoadSession(sessionOptions, _ttsDir, _ttsMeta.files.local_decoder);

            if (!string.IsNullOrEmpty(_ttsMeta.files.local_greedy_frame))
                _localGreedyFrameSession = LoadSession(sessionOptions, _ttsDir, _ttsMeta.files.local_greedy_frame);
            if (!string.IsNullOrEmpty(_ttsMeta.files.local_fixed_sampled_frame))
                _localFixedSampledFrameSession = LoadSession(sessionOptions, _ttsDir, _ttsMeta.files.local_fixed_sampled_frame);
            if (!string.IsNullOrEmpty(_ttsMeta.files.local_cached_step))
                _localCachedStepSession = LoadSession(sessionOptions, _ttsDir, _ttsMeta.files.local_cached_step);

            _codecEncodeSession = LoadSession(sessionOptions, _codecDir, _codecMeta.files.encode);
            _codecDecodeSession = LoadSession(sessionOptions, _codecDir, _codecMeta.files.decode_full);
            _codecDecodeStepSession = LoadSession(sessionOptions, _codecDir, _codecMeta.files.decode_step);

            InitializeStreamingState();

            Debug.Log("[OnnxRuntime] All inference sessions initialized");
        }

        private SessionOptions CreateSessionOptions(int threadCount)
        {
            return new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 1,
                IntraOpNumThreads = threadCount
            };
        }

        private InferenceSession LoadSession(SessionOptions options, string dir, string fileName)
        {
            string path = Path.Combine(dir, fileName);
            if (!File.Exists(path))
            {
                Debug.LogWarning($"[OnnxRuntime] Model file not found: {path}");
                return null;
            }
            return new InferenceSession(path, options);
        }

        private void InitializeStreamingState()
        {
            _streamingInputs = new Dictionary<string, NamedOnnxValue>();

            if (_codecMeta.streaming_decode?.transformer_offsets != null)
            {
                foreach (var spec in _codecMeta.streaming_decode.transformer_offsets)
                {
                    _streamingInputs[spec.input_name] = NamedOnnxValue.CreateFromTensor(spec.input_name, CreateZeroTensor<int>(spec.shape));
                }
            }

            if (_codecMeta.streaming_decode?.attention_caches != null)
            {
                foreach (var spec in _codecMeta.streaming_decode.attention_caches)
                {
                    _streamingInputs[spec.offset_input_name] = NamedOnnxValue.CreateFromTensor(spec.offset_input_name, CreateZeroTensor<int>(spec.offset_shape));
                    _streamingInputs[spec.cached_keys_input_name] = NamedOnnxValue.CreateFromTensor(spec.cached_keys_input_name, CreateZeroTensor<float>(spec.cache_shape));
                    _streamingInputs[spec.cached_values_input_name] = NamedOnnxValue.CreateFromTensor(spec.cached_values_input_name, CreateZeroTensor<float>(spec.cache_shape));
                    _streamingInputs[spec.cached_positions_input_name] = NamedOnnxValue.CreateFromTensor(spec.cached_positions_input_name, CreateZeroTensor<int>(spec.positions_shape));
                }
            }
        }

        private DenseTensor<T> CreateZeroTensor<T>(int[] shape) where T : struct
        {
            int totalLength = 1;
            foreach (int dim in shape) totalLength *= dim;
            T[] data = new T[totalLength];
            return new DenseTensor<T>(data, shape);
        }

        /// <summary>
        /// 把扁平 KV cache 还原成 [1, seq, heads, headDim] 四维张量。
        /// decode_step / local_cached_step 的 past_* 输入都是 4 维，
        /// 用 [1, N] 二维会直接被 ORT 判为 shape mismatch。
        /// </summary>
        private DenseTensor<float> MakeKvTensor(float[] data, int heads, int headDim)
        {
            int stride = Math.Max(1, heads * headDim);
            int seq = data.Length / stride;
            return new DenseTensor<float>(data, new[] { 1, seq, heads, headDim });
        }

        /// <summary>
        /// 读取 int32 标志位输出。ONNX 导出的 should_continue 是 INT32 而非 BOOL，
        /// 用 AsTensor&lt;bool&gt;() 会静默返回 null 并在后续解引用时抛 NullReferenceException。
        /// </summary>
        private static bool ReadFlag(IReadOnlyCollection<DisposableNamedOnnxValue> results, string name)
        {
            var value = results.FirstOrDefault(r => r.Name == name);
            if (value == null)
                throw new InvalidOperationException($"[OnnxRuntime] Output '{name}' not found");

            var intTensor = value.AsTensor<int>();
            if (intTensor != null)
                return intTensor.GetValue(0) != 0;

            var longTensor = value.AsTensor<long>();
            if (longTensor != null)
                return longTensor.GetValue(0) != 0;

            var boolTensor = value.AsTensor<bool>();
            if (boolTensor != null)
                return boolTensor.GetValue(0);

            throw new InvalidOperationException($"[OnnxRuntime] Unsupported tensor type for output '{name}'");
        }

        public (float[] globalHidden, Dictionary<string, float[]> pastStates, int pastValidLength) Prefill(
            int[,,] inputIds, int[,] attentionMask)
        {
            int seqLen = inputIds.GetLength(1);
            int rowWidth = inputIds.GetLength(2);

            int[] flatInput = Flatten3D(inputIds);
            int[] flatMask = Flatten2D(attentionMask);

            var inputTensor = new DenseTensor<int>(flatInput, new[] { 1, seqLen, rowWidth });
            var maskTensor = new DenseTensor<int>(flatMask, new[] { 1, seqLen });

            var inputs = new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor)
            };

            using var results = _prefillSession.Run(inputs);
            var outputNames = _prefillSession.OutputNames;

            // global_hidden: [1, seqLen, hidden_size] → take last timestep
            var globalHiddenTensor = results.First(r => r.Name == "global_hidden").AsTensor<float>();
            int hiddenSize = globalHiddenTensor.Dimensions[globalHiddenTensor.Dimensions.Length - 1];
            float[] globalHidden = TensorTail(globalHiddenTensor, hiddenSize);

            var pastStates = new Dictionary<string, float[]>();
            foreach (string name in outputNames)
            {
                if (name != "global_hidden")
                {
                    var tensor = results.First(r => r.Name == name).AsTensor<float>();
                    pastStates[name.Replace("present_", "past_")] = TensorToArray(tensor);
                }
            }

            return (globalHidden, pastStates, seqLen);
        }

        public (float[] globalHidden, Dictionary<string, float[]> newPastStates) Decode(
            int[,,] inputIds, int pastValidLength, Dictionary<string, float[]> pastStates)
        {
            int rowWidth = inputIds.GetLength(2);
            int[] flatInput = Flatten3D(inputIds);

            var inputTensor = new DenseTensor<int>(flatInput, new[] { 1, 1, rowWidth });
            var pastValidTensor = new DenseTensor<int>(new[] { pastValidLength }, new[] { 1 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
                NamedOnnxValue.CreateFromTensor("past_valid_lengths", pastValidTensor)
            };

            foreach (var kvp in pastStates)
            {
                var pastTensor = MakeKvTensor(kvp.Value, _globalHeads, _globalHeadDim);
                inputs.Add(NamedOnnxValue.CreateFromTensor(kvp.Key, pastTensor));
            }

            using var results = _decodeSession.Run(inputs);
            var outputNames = _decodeSession.OutputNames;

            // global_hidden: [1, 1, hidden_size] → take last timestep
            var globalHiddenTensor = results.First(r => r.Name == "global_hidden").AsTensor<float>();
            int hiddenSize = globalHiddenTensor.Dimensions[globalHiddenTensor.Dimensions.Length - 1];
            float[] globalHidden = TensorTail(globalHiddenTensor, hiddenSize);

            var newPastStates = new Dictionary<string, float[]>();
            foreach (string name in outputNames)
            {
                if (name != "global_hidden")
                {
                    var tensor = results.First(r => r.Name == name).AsTensor<float>();
                    newPastStates[name.Replace("present_", "past_")] = TensorToArray(tensor);
                }
            }

            return (globalHidden, newPastStates);
        }

        public (float[] textLogits, float[] audioLogits) LocalDecoder(
            float[] globalHidden, int textTokenId, int[] audioPrefix)
        {
            int audioPad = _manifest.tts_config.audio_pad_token_id;
            var paddedPrefix = new int[_nVq - 1];
            Array.Fill(paddedPrefix, audioPad);
            Array.Copy(audioPrefix, paddedPrefix, Math.Min(audioPrefix.Length, _nVq - 1));

            // global_hidden: [1, hidden_size]
            var globalHiddenTensor = new DenseTensor<float>(globalHidden, new[] { 1, globalHidden.Length });
            var textTokenTensor = new DenseTensor<int>(new[] { textTokenId }, new[] { 1 });
            var audioPrefixTensor = new DenseTensor<int>(paddedPrefix, new[] { 1, _nVq - 1 });

            var inputs = new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("global_hidden", globalHiddenTensor),
                NamedOnnxValue.CreateFromTensor("text_token_id", textTokenTensor),
                NamedOnnxValue.CreateFromTensor("audio_prefix_token_ids", audioPrefixTensor)
            };

            using var results = _localDecoderSession.Run(inputs);

            float[] textLogits = TensorToArray(results.First(r => r.Name == "text_logits").AsTensor<float>());
            float[] audioLogits = TensorToArray(results.First(r => r.Name == "audio_logits").AsTensor<float>());

            return (textLogits, audioLogits);
        }

        /// <summary>
        /// 构建 repetition_seen_mask 张量，复用同一块缓冲区。
        /// 形状 [1, nVq, codebookSize]，每帧调用一次；每次新建会在长文本合成时
        /// 产生大量 LOH 垃圾（16×1024×4B = 64KB/帧），因此改为原地清零 + 重填。
        /// </summary>
        private DenseTensor<int> BuildRepetitionMask(List<HashSet<int>> previousTokenSetsByChannel, int codebookSize)
        {
            int total = _nVq * codebookSize;
            if (_repetitionMaskBuffer == null || _repetitionMaskBuffer.Length != total)
            {
                _repetitionMaskBuffer = new int[total];
                _repetitionMaskTensor = new DenseTensor<int>(_repetitionMaskBuffer, new[] { 1, _nVq, codebookSize });
            }
            else
            {
                Array.Clear(_repetitionMaskBuffer, 0, total);
            }

            int channels = Math.Min(previousTokenSetsByChannel.Count, _nVq);
            for (int ch = 0; ch < channels; ch++)
            {
                int channelBase = ch * codebookSize;
                foreach (int tokenId in previousTokenSetsByChannel[ch])
                {
                    if (tokenId >= 0 && tokenId < codebookSize)
                        _repetitionMaskBuffer[channelBase + tokenId] = 1;
                }
            }

            return _repetitionMaskTensor;
        }

        public (bool shouldContinue, int[] frameTokenIds) LocalGreedyFrame(
            float[] globalHidden, List<HashSet<int>> previousTokenSetsByChannel, float repetitionPenalty)
        {
            if (_localGreedyFrameSession == null)
                throw new InvalidOperationException("LocalGreedyFrame session not loaded");

            var globalHiddenTensor = new DenseTensor<float>(globalHidden, new[] { 1, globalHidden.Length });
            var maskTensor = BuildRepetitionMask(previousTokenSetsByChannel, _codebookSize);
            var penaltyTensor = new DenseTensor<float>(new[] { repetitionPenalty }, new[] { 1 });

            var inputs = new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("global_hidden", globalHiddenTensor),
                NamedOnnxValue.CreateFromTensor("repetition_seen_mask", maskTensor),
                NamedOnnxValue.CreateFromTensor("repetition_penalty", penaltyTensor)
            };

            using var results = _localGreedyFrameSession.Run(inputs);

            bool shouldContinue = ReadFlag(results, "should_continue");
            int[] frameTokenIds = TensorToArray(results.First(r => r.Name == "frame_token_ids").AsTensor<int>());

            return (shouldContinue, frameTokenIds);
        }

        public (bool shouldContinue, int[] frameTokenIds) LocalFixedSampledFrame(
            float[] globalHidden, List<HashSet<int>> previousTokenSetsByChannel, System.Random rng)
        {
            if (_localFixedSampledFrameSession == null)
                throw new InvalidOperationException("LocalFixedSampledFrame session not loaded");

            var maskTensor = BuildRepetitionMask(previousTokenSetsByChannel, _codebookSize);

            // 随机数消耗顺序必须与 Python 参考实现一致：先 assistant，再 audio。
            // 每帧 frame_s 先从 rng 取 1 个 [1] 给 assistant_random_u，再取 N_VQ 个 [1,N_VQ]
            // 给 audio_random_u。顺序反了会让模型收到完全错的随机值，should_continue 决策都不同。
            float assistantRandom = (float)rng.NextDouble();

            // 随机数缓冲同样复用，避免每帧的 LINQ 分配。
            if (_audioRandomBuffer == null || _audioRandomBuffer.Length != _nVq)
            {
                _audioRandomBuffer = new float[_nVq];
                _audioRandomTensor = new DenseTensor<float>(_audioRandomBuffer, new[] { 1, _nVq });
            }
            for (int i = 0; i < _nVq; i++)
                _audioRandomBuffer[i] = (float)rng.NextDouble();

            var globalHiddenTensor = new DenseTensor<float>(globalHidden, new[] { 1, globalHidden.Length });
            var assistantRandomTensor = new DenseTensor<float>(new[] { assistantRandom }, new[] { 1 });
            var audioRandomTensor = _audioRandomTensor;

            var inputs = new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("global_hidden", globalHiddenTensor),
                NamedOnnxValue.CreateFromTensor("repetition_seen_mask", maskTensor),
                NamedOnnxValue.CreateFromTensor("assistant_random_u", assistantRandomTensor),
                NamedOnnxValue.CreateFromTensor("audio_random_u", audioRandomTensor)
            };

            using var results = _localFixedSampledFrameSession.Run(inputs);

            bool shouldContinue = ReadFlag(results, "should_continue");
            int[] frameTokenIds = TensorToArray(results.First(r => r.Name == "frame_token_ids").AsTensor<int>());

            return (shouldContinue, frameTokenIds);
        }

        public (float[] textLogits, float[] audioLogits, Dictionary<string, float[]> nextPast) LocalCachedStep(
            float[] globalHidden, int textTokenId, int audioTokenId,
            int channelIndex, int stepType, int pastValidLengths,
            Dictionary<string, float[]> localPastByName)
        {
            if (_localCachedStepSession == null)
                throw new InvalidOperationException("LocalCachedStep session not loaded");

            var globalHiddenTensor = new DenseTensor<float>(globalHidden, new[] { 1, globalHidden.Length });
            var textTokenTensor = new DenseTensor<int>(new[] { textTokenId }, new[] { 1 });
            var audioTokenTensor = new DenseTensor<int>(new[] { audioTokenId }, new[] { 1 });
            var channelTensor = new DenseTensor<int>(new[] { channelIndex }, new[] { 1 });
            var stepTypeTensor = new DenseTensor<int>(new[] { stepType }, new[] { 1 });
            var pastValidTensor = new DenseTensor<int>(new[] { pastValidLengths }, new[] { 1 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("global_hidden", globalHiddenTensor),
                NamedOnnxValue.CreateFromTensor("text_token_id", textTokenTensor),
                NamedOnnxValue.CreateFromTensor("audio_token_id", audioTokenTensor),
                NamedOnnxValue.CreateFromTensor("channel_index", channelTensor),
                NamedOnnxValue.CreateFromTensor("step_type", stepTypeTensor),
                NamedOnnxValue.CreateFromTensor("past_valid_lengths", pastValidTensor)
            };

            foreach (var kvp in localPastByName)
            {
                var pastTensor = MakeKvTensor(kvp.Value, _localHeads, _localHeadDim);
                inputs.Add(NamedOnnxValue.CreateFromTensor(kvp.Key, pastTensor));
            }

            using var results = _localCachedStepSession.Run(inputs);

            float[] textLogits = TensorToArray(results.First(r => r.Name == "text_logits").AsTensor<float>());
            float[] audioLogits = TensorToArray(results.First(r => r.Name == "audio_logits").AsTensor<float>());

            var nextPast = new Dictionary<string, float[]>();
            foreach (string name in _ttsMeta.onnx.local_cached_output_names.Skip(2))
            {
                var tensor = results.First(r => r.Name == name).AsTensor<float>();
                nextPast[name.Replace("local_present_", "local_past_")] = TensorToArray(tensor);
            }

            return (textLogits, audioLogits, nextPast);
        }

        public float[] SliceAudioChannelLogits(float[] audioLogits, int channelIndex, int nVq, int codebookSize)
        {
            float[] result = new float[codebookSize];
            int start = channelIndex * codebookSize;
            if (start + codebookSize <= audioLogits.Length)
                Array.Copy(audioLogits, start, result, 0, codebookSize);
            return result;
        }

        public List<int[]> CodecEncode(float[] waveform, int sampleRate)
        {
            return new List<int[]>();
        }

        public (float[][] channelArrays, int audioLength) CodecDecode(List<int[]> generatedFrames)
        {
            if (generatedFrames.Count == 0)
                return (Array.Empty<float[]>(), 0);

            int numQuantizers = _codecMeta.codec_config.num_quantizers;
            int framesCount = generatedFrames.Count;
            var audioCodesData = new int[1 * framesCount * numQuantizers];

            for (int f = 0; f < framesCount; f++)
            {
                for (int q = 0; q < numQuantizers && q < generatedFrames[f].Length; q++)
                {
                    audioCodesData[f * numQuantizers + q] = generatedFrames[f][q];
                }
            }

            var audioCodesTensor = new DenseTensor<int>(audioCodesData, new[] { 1, framesCount, numQuantizers });
            var lengthsTensor = new DenseTensor<int>(new[] { framesCount }, new[] { 1 });

            var inputs = new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("audio_codes", audioCodesTensor),
                NamedOnnxValue.CreateFromTensor("audio_code_lengths", lengthsTensor)
            };

            using var results = _codecDecodeSession.Run(inputs);

            var audioTensor = results.First(r => r.Name == "audio").AsTensor<float>();
            var lengthsOutput = results.First(r => r.Name == "audio_lengths").AsTensor<int>();
            int audioLength = TensorToArray(lengthsOutput)[0];

            // audio 形状为 [batch, channels, audio_length]（通道优先），不是交错排列
            var audioData = TensorToArray(audioTensor);
            var dims = audioTensor.Dimensions;
            int channels = dims.Length >= 3 ? dims[1] : _codecMeta.codec_config.channels;
            int frameStride = dims.Length >= 3 ? dims[dims.Length - 1] : audioLength;
            audioLength = Math.Min(audioLength, frameStride);

            var channelArrays = new float[channels][];
            for (int c = 0; c < channels; c++)
            {
                channelArrays[c] = new float[audioLength];
                Array.Copy(audioData, c * frameStride, channelArrays[c], 0, audioLength);
            }

            return (channelArrays, audioLength);
        }

        public (float[] audio, int audioLength) CodecDecodeStep(List<int[]> frameRows)
        {
            if (frameRows.Count == 0)
                return (Array.Empty<float>(), 0);

            int numQuantizers = _codecMeta.codec_config.num_quantizers;
            int framesCount = frameRows.Count;
            var audioCodesData = new int[1 * framesCount * numQuantizers];

            for (int f = 0; f < framesCount; f++)
            {
                for (int q = 0; q < numQuantizers && q < frameRows[f].Length; q++)
                {
                    audioCodesData[f * numQuantizers + q] = frameRows[f][q];
                }
            }

            var audioCodesTensor = new DenseTensor<int>(audioCodesData, new[] { 1, framesCount, numQuantizers });
            var lengthsTensor = new DenseTensor<int>(new[] { framesCount }, new[] { 1 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("audio_codes", audioCodesTensor),
                NamedOnnxValue.CreateFromTensor("audio_code_lengths", lengthsTensor)
            };

            // 添加 streaming states
            foreach (var kvp in _streamingInputs)
            {
                inputs.Add(kvp.Value);
            }

            using var results = _codecDecodeStepSession.Run(inputs);

            var audioTensor = results.First(r => r.Name == "audio").AsTensor<float>();
            var lengthsOutput = results.First(r => r.Name == "audio_lengths").AsTensor<int>();
            int audioLength = TensorToArray(lengthsOutput)[0];

            // 输出是 [batch, channels, audio_length]（平面排列），调用方期望交错排列
            var planar = TensorToArray(audioTensor);
            var dims = audioTensor.Dimensions;
            int channels = dims.Length >= 3 ? dims[1] : _codecMeta.codec_config.channels;
            int frameStride = dims.Length >= 3 ? dims[dims.Length - 1] : audioLength;
            audioLength = Math.Min(audioLength, frameStride);

            float[] audio = new float[audioLength * channels];
            for (int c = 0; c < channels; c++)
                for (int s = 0; s < audioLength; s++)
                    audio[s * channels + c] = planar[c * frameStride + s];

            UpdateStreamingStates(results);

            return (audio, audioLength);
        }

        private void UpdateStreamingStates(IReadOnlyCollection<DisposableNamedOnnxValue> results)
        {
            if (_codecMeta.streaming_decode?.transformer_offsets != null)
            {
                foreach (var spec in _codecMeta.streaming_decode.transformer_offsets)
                {
                    string outName = spec.output_name;
                    var tensor = results.FirstOrDefault(r => r.Name == outName)?.AsTensor<int>();
                    if (tensor != null)
                        _streamingInputs[spec.input_name] = NamedOnnxValue.CreateFromTensor(spec.input_name, CloneTensor(tensor));
                }
            }

            if (_codecMeta.streaming_decode?.attention_caches != null)
            {
                foreach (var spec in _codecMeta.streaming_decode.attention_caches)
                {
                    var offsetTensor = results.FirstOrDefault(r => r.Name == spec.offset_output_name)?.AsTensor<int>();
                    var keysTensor = results.FirstOrDefault(r => r.Name == spec.cached_keys_output_name)?.AsTensor<float>();
                    var valuesTensor = results.FirstOrDefault(r => r.Name == spec.cached_values_output_name)?.AsTensor<float>();
                    var positionsTensor = results.FirstOrDefault(r => r.Name == spec.cached_positions_output_name)?.AsTensor<int>();

                    if (offsetTensor != null)
                        _streamingInputs[spec.offset_input_name] = NamedOnnxValue.CreateFromTensor(spec.offset_input_name, CloneTensor(offsetTensor));
                    if (keysTensor != null)
                        _streamingInputs[spec.cached_keys_input_name] = NamedOnnxValue.CreateFromTensor(spec.cached_keys_input_name, CloneTensor(keysTensor));
                    if (valuesTensor != null)
                        _streamingInputs[spec.cached_values_input_name] = NamedOnnxValue.CreateFromTensor(spec.cached_values_input_name, CloneTensor(valuesTensor));
                    if (positionsTensor != null)
                        _streamingInputs[spec.cached_positions_input_name] = NamedOnnxValue.CreateFromTensor(spec.cached_positions_input_name, CloneTensor(positionsTensor));
                }
            }
        }

        private DenseTensor<T> CloneTensor<T>(Tensor<T> tensor) where T : struct
        {
            // 走 TensorToArray 的 span 快路径，避免逐元素 GetValue。
            // 流式解码每帧都要克隆 12 组 attention cache，这里是热点。
            return new DenseTensor<T>(TensorToArray(tensor), tensor.Dimensions.ToArray());
        }

        public void CodecDecodeStepReset()
        {
            InitializeStreamingState();
        }

        #region Utility Methods

        private int[] Flatten3D(int[,,] array)
        {
            int d0 = array.GetLength(0), d1 = array.GetLength(1), d2 = array.GetLength(2);
            int[] result = new int[d0 * d1 * d2];
            int idx = 0;
            for (int i = 0; i < d0; i++)
                for (int j = 0; j < d1; j++)
                    for (int k = 0; k < d2; k++)
                        result[idx++] = array[i, j, k];
            return result;
        }

        private int[] Flatten2D(int[,] array)
        {
            int d0 = array.GetLength(0), d1 = array.GetLength(1);
            int[] result = new int[d0 * d1];
            int idx = 0;
            for (int i = 0; i < d0; i++)
                for (int j = 0; j < d1; j++)
                    result[idx++] = array[i, j];
            return result;
        }

        /// <summary>
        /// 把张量内容拷成托管数组。
        ///
        /// 性能说明：ORT 返回的张量实际类型都是 DenseTensor&lt;T&gt;，其 Buffer 是连续内存，
        /// 可以直接 Span.CopyTo 走 memcpy。原实现用 tensor.GetValue(i) 逐元素读，
        /// 每次都要过虚方法 + 边界检查 + ArrayUtilities.GetIndex 计算多维索引；
        /// 单次 decode step 要搬 24 个 KV 张量（约 415 万元素），60 步累计约 2.5 亿次虚调用，
        /// 是 C# 侧最大的单项开销。这里优先走 span 路径，非 DenseTensor 时才回退逐元素。
        /// </summary>
        private static T[] TensorToArray<T>(Tensor<T> tensor) where T : struct
        {
            int length = (int)tensor.Length;

            if (tensor is DenseTensor<T> dense)
            {
                var span = dense.Buffer.Span;
                // Buffer 理论上可能大于逻辑长度，按 Length 截断后再拷。
                if (span.Length >= length)
                    return span.Slice(0, length).ToArray();
            }

            T[] result = new T[length];
            for (int i = 0; i < length; i++)
                result[i] = tensor.GetValue(i);
            return result;
        }

        /// <summary>
        /// 从张量尾部取 count 个元素（用于取 global_hidden 的最后一个时间步）。
        /// 同样优先走连续内存拷贝，避免逐元素 GetValue。
        /// </summary>
        private static float[] TensorTail(Tensor<float> tensor, int count)
        {
            int length = (int)tensor.Length;
            int offset = Math.Max(0, length - count);
            int actual = Math.Min(count, length);

            if (tensor is DenseTensor<float> dense)
            {
                var span = dense.Buffer.Span;
                if (span.Length >= length)
                {
                    float[] fast = new float[count];
                    span.Slice(offset, actual).CopyTo(fast);
                    return fast;
                }
            }

            float[] result = new float[count];
            for (int i = 0; i < actual; i++)
                result[i] = tensor.GetValue(offset + i);
            return result;
        }

        #endregion

        public void Dispose()
        {
            if (!_disposed)
            {
                _prefillSession?.Dispose();
                _decodeSession?.Dispose();
                _localDecoderSession?.Dispose();
                _localGreedyFrameSession?.Dispose();
                _localFixedSampledFrameSession?.Dispose();
                _localCachedStepSession?.Dispose();
                _codecEncodeSession?.Dispose();
                _codecDecodeSession?.Dispose();
                _codecDecodeStepSession?.Dispose();

                _disposed = true;
            }
        }
    }
}
