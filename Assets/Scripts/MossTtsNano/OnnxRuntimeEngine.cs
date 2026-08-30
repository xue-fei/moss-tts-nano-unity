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

        private Dictionary<string, NamedOnnxValue> _streamingInputs;

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

            // Resolve codec_meta path
            string codecMetaPath = Path.Combine(_modelDir, _manifest.model_files.codec_meta);
            if (!File.Exists(codecMetaPath))
            {
                // Fallback: try relative to manifest dir without ".."
                codecMetaPath = Path.Combine(_modelDir, "MOSS-Audio-Tokenizer-Nano-ONNX", "codec_browser_onnx_meta.json");
            }
            string codecMetaJson = File.ReadAllText(codecMetaPath);
            _codecMeta = JsonConvert.DeserializeObject<CodecModelMeta>(codecMetaJson);
            _codecDir = Path.GetDirectoryName(codecMetaPath);

            Debug.Log($"[OnnxRuntime] Manifest loaded: {_manifest.builtin_voices?.Length} voices, nVq={_nVq}, codebook={_codebookSize}");
            Debug.Log($"[OnnxRuntime] prompt_templates: {(_manifest.prompt_templates != null ? "OK" : "NULL")}");
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
            float[] globalHidden = new float[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
                globalHidden[i] = globalHiddenTensor.GetValue((int)(globalHiddenTensor.Length - hiddenSize + i));

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
                var pastTensor = new DenseTensor<float>(kvp.Value, new[] { 1, kvp.Value.Length });
                inputs.Add(NamedOnnxValue.CreateFromTensor(kvp.Key, pastTensor));
            }

            using var results = _decodeSession.Run(inputs);
            var outputNames = _decodeSession.OutputNames;

            // global_hidden: [1, 1, hidden_size] → take last timestep
            var globalHiddenTensor = results.First(r => r.Name == "global_hidden").AsTensor<float>();
            int hiddenSize = globalHiddenTensor.Dimensions[globalHiddenTensor.Dimensions.Length - 1];
            float[] globalHidden = new float[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
                globalHidden[i] = globalHiddenTensor.GetValue((int)(globalHiddenTensor.Length - hiddenSize + i));

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

        public (bool shouldContinue, int[] frameTokenIds) LocalGreedyFrame(
            float[] globalHidden, List<HashSet<int>> previousTokenSetsByChannel, float repetitionPenalty)
        {
            if (_localGreedyFrameSession == null)
                throw new InvalidOperationException("LocalGreedyFrame session not loaded");

            int audioCodebookSize = _codebookSize;

            var maskData = new int[1 * _nVq * audioCodebookSize];
            for (int ch = 0; ch < previousTokenSetsByChannel.Count; ch++)
            {
                foreach (int tokenId in previousTokenSetsByChannel[ch])
                {
                    if (tokenId >= 0 && tokenId < audioCodebookSize)
                        maskData[ch * audioCodebookSize + tokenId] = 1;
                }
            }

            var globalHiddenTensor = new DenseTensor<float>(globalHidden, new[] { 1, globalHidden.Length });
            var maskTensor = new DenseTensor<int>(maskData, new[] { 1, _nVq, audioCodebookSize });
            var penaltyTensor = new DenseTensor<float>(new[] { repetitionPenalty }, new[] { 1 });

            var inputs = new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("global_hidden", globalHiddenTensor),
                NamedOnnxValue.CreateFromTensor("repetition_seen_mask", maskTensor),
                NamedOnnxValue.CreateFromTensor("repetition_penalty", penaltyTensor)
            };

            using var results = _localGreedyFrameSession.Run(inputs);

            bool shouldContinue = TensorToArray(results.First(r => r.Name == "should_continue").AsTensor<bool>())[0];
            int[] frameTokenIds = TensorToArray(results.First(r => r.Name == "frame_token_ids").AsTensor<int>());

            return (shouldContinue, frameTokenIds);
        }

        public (bool shouldContinue, int[] frameTokenIds) LocalFixedSampledFrame(
            float[] globalHidden, List<HashSet<int>> previousTokenSetsByChannel, System.Random rng)
        {
            if (_localFixedSampledFrameSession == null)
                throw new InvalidOperationException("LocalFixedSampledFrame session not loaded");

            int audioCodebookSize = _codebookSize;

            var maskData = new int[1 * _nVq * audioCodebookSize];
            for (int ch = 0; ch < previousTokenSetsByChannel.Count; ch++)
            {
                foreach (int tokenId in previousTokenSetsByChannel[ch])
                {
                    if (tokenId >= 0 && tokenId < audioCodebookSize)
                        maskData[ch * audioCodebookSize + tokenId] = 1;
                }
            }

            float assistantRandom = (float)rng.NextDouble();
            float[] audioRandom = Enumerable.Range(0, _nVq).Select(_ => (float)rng.NextDouble()).ToArray();

            var globalHiddenTensor = new DenseTensor<float>(globalHidden, new[] { 1, globalHidden.Length });
            var maskTensor = new DenseTensor<int>(maskData, new[] { 1, _nVq, audioCodebookSize });
            var assistantRandomTensor = new DenseTensor<float>(new[] { assistantRandom }, new[] { 1 });
            var audioRandomTensor = new DenseTensor<float>(audioRandom, new[] { 1, _nVq });

            var inputs = new NamedOnnxValue[]
            {
                NamedOnnxValue.CreateFromTensor("global_hidden", globalHiddenTensor),
                NamedOnnxValue.CreateFromTensor("repetition_seen_mask", maskTensor),
                NamedOnnxValue.CreateFromTensor("assistant_random_u", assistantRandomTensor),
                NamedOnnxValue.CreateFromTensor("audio_random_u", audioRandomTensor)
            };

            using var results = _localFixedSampledFrameSession.Run(inputs);

            bool shouldContinue = TensorToArray(results.First(r => r.Name == "should_continue").AsTensor<bool>())[0];
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
                var pastTensor = new DenseTensor<float>(kvp.Value, new[] { 1, kvp.Value.Length });
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

            int channels = _codecMeta.codec_config.channels;
            var audioData = TensorToArray(audioTensor);
            var channelArrays = new float[channels][];

            for (int c = 0; c < channels; c++)
            {
                channelArrays[c] = new float[audioLength];
                for (int s = 0; s < audioLength; s++)
                {
                    channelArrays[c][s] = audioData[s * channels + c];
                }
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

            float[] audio = TensorToArray(audioTensor);

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
            T[] data = new T[tensor.Length];
            for (int i = 0; i < tensor.Length; i++)
                data[i] = tensor.GetValue(i);
            return new DenseTensor<T>(data, tensor.Dimensions.ToArray());
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

        private T[] TensorToArray<T>(Tensor<T> tensor) where T : struct
        {
            T[] result = new T[tensor.Length];
            for (int i = 0; i < tensor.Length; i++)
                result[i] = tensor.GetValue(i);
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
