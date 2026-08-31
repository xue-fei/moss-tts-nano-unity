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
            _outputDir = Path.Combine(Application.dataPath, "MossTtsOutput");
            Directory.CreateDirectory(_outputDir);
            _tokenCache = new Dictionary<string, int>();
            _voiceCache = new Dictionary<string, List<int[]>>();

            // 加载 SentencePiece 分词器
            InitializeTokenizer(modelDir);
        }

        private void InitializeTokenizer(string modelDir)
        {
            // 尝试多个路径查找 tokenizer 词汇表
            string[] candidatePaths = {
                Path.Combine(_modelDir, "tokenizer_vocab_parallel.json"),
                Path.Combine(_modelDir, "MOSS-TTS-Nano-100M-ONNX", "tokenizer_vocab_parallel.json"),
                Path.Combine(_modelDir, "tokenizer.model")
            };

            foreach (var path in candidatePaths)
            {
                string fullPath = Path.GetFullPath(path);
                if (File.Exists(fullPath))
                {
                    try
                    {
                        _tokenizer = new SentencePieceTokenizer(fullPath);
                        Debug.Log($"[OnnxTtsRuntime] Tokenizer loaded from {fullPath}");
                        return;
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning($"[OnnxTtsRuntime] Failed to load tokenizer from {fullPath}: {e.Message}");
                    }
                }
            }

            // 回退到简单 Unicode 分词器（仅用于测试）
            Debug.LogWarning("[OnnxTtsRuntime] SentencePiece tokenizer not found, falling back to simple Unicode tokenizer");
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
                if (generatedFrames.Count == 0)
                {
                    Debug.LogWarning($"[OnnxTtsRuntime] No audio frames generated for '{text}' (model stopped at step 0)");
                    return (Array.Empty<float>(), generatedFrames);
                }
                var (channelArrays, audioLength) = _engine.CodecDecode(generatedFrames);
                waveform = MergeAudioChannels(channelArrays, audioLength);
                Debug.Log($"[OnnxTtsRuntime] Chunk done: {generatedFrames.Count} frames -> {audioLength} samples/ch");
                return (waveform, generatedFrames);
            }

            // 流式合成
            var emittedChunks = new List<float[]>();
            int emittedSamplesTotal = 0;
            float? firstAudioEmittedAt = null;
            var pendingDecodeFrames = new List<int[]>();

            // 每个 chunk 独立的流式解码状态，避免跨 chunk 缓存串味
            _engine.CodecDecodeStepReset();

            void OnFrame(List<int[]> frames, int step, int[] frame)
            {
                // 只解码尚未消费的新帧，否则每次回调都会把历史帧重复解码一遍
                pendingDecodeFrames.Add(frame);

                int decodeBudget = Math.Max(1, ResolveStreamDecodeFrameBudget(
                    emittedSamplesTotal, _codecMeta.codec_config.sample_rate, firstAudioEmittedAt));

                if (pendingDecodeFrames.Count < decodeBudget)
                    return;

                var (audio, audioLength) = _engine.CodecDecodeStep(pendingDecodeFrames);
                pendingDecodeFrames.Clear();

                if (audioLength > 0)
                {
                    if (!firstAudioEmittedAt.HasValue)
                        firstAudioEmittedAt = Time.realtimeSinceStartup;

                    emittedSamplesTotal += audioLength;
                    emittedChunks.Add(audio);
                }
            }

            generatedFrames = GenerateAudioFrames((inputIds, attentionMask), OnFrame);

            // 刷新残留的未解码帧（不是重新解码全部帧）
            if (pendingDecodeFrames.Count > 0)
            {
                var (audio, audioLength) = _engine.CodecDecodeStep(pendingDecodeFrames);
                pendingDecodeFrames.Clear();
                if (audioLength > 0)
                    emittedChunks.Add(audio);
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

                void OnFrame(List<int[]> frames, int step, int[] frame)
                {
                    pendingDecodeFrames.Add(frame);
                    int decodeBudget = ResolveStreamDecodeFrameBudget(emittedSamplesTotal, sampleRate, firstAudioEmittedAt);

                    if (pendingDecodeFrames.Count >= Math.Max(1, decodeBudget))
                    {
                        var (audio, audioLength) = _engine.CodecDecodeStep(pendingDecodeFrames);
                        pendingDecodeFrames.Clear();

                        if (audioLength > 0)
                        {
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
                    }
                }

                GenerateAudioFrames((inputIds, attentionMask), OnFrame);
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
