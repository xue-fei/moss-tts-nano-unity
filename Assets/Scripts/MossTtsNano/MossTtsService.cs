using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// MOSS-TTS-Nano 高级服务 - 封装运行时管理和合成操作
    /// 对应 Python 的 NanoTTSService / OnnxNanoTTSServiceAdapter
    /// </summary>
    public class MossTtsService : IDisposable
    {
        private OnnxTtsRuntime _runtime;
        private readonly string _modelDir;
        private readonly string _outputDir;
        private readonly int _threadCount;
        private readonly string _executionProvider;
        private readonly object _lock = new object();
        private bool _isLoaded;
        private bool _disposed;

        public bool IsLoaded => _isLoaded;
        public ModelManifest Manifest => _runtime?.Manifest;

        public MossTtsService(
            string modelDir,
            string outputDir = null,
            int threadCount = 4,
            string executionProvider = "cpu")
        {
            _modelDir = modelDir;
            _outputDir = outputDir ?? Path.Combine(Application.persistentDataPath, "MossTtsOutput");
            _threadCount = threadCount;
            _executionProvider = executionProvider;
        }

        /// <summary>
        /// 加载模型
        /// </summary>
        public void LoadModel()
        {
            if (_isLoaded) return;

            lock (_lock)
            {
                if (_isLoaded) return;

                _runtime = new OnnxTtsRuntime(
                    _modelDir,
                    _threadCount,
                    executionProvider: _executionProvider,
                    outputDir: _outputDir);

                _isLoaded = true;
                Debug.Log($"[MossTts] Model loaded from {_modelDir}");
            }
        }

        /// <summary>
        /// 预热模型
        /// </summary>
        public void Warmup()
        {
            EnsureLoaded();
            // 使用简单文本预热
            _runtime.Synthesize(
                "你好",
                voice: "Junhao",
                maxNewFrames: 96,
                doSample: false);
        }

        /// <summary>
        /// 合成语音
        /// </summary>
        public SynthesisResult Synthesize(
            string text,
            string voice = null,
            string promptAudioPath = null,
            string outputPath = null,
            int? maxNewFrames = null,
            int voiceCloneMaxTextTokens = 75,
            bool doSample = true,
            string sampleMode = "fixed",
            bool streaming = false,
            int? seed = null)
        {
            EnsureLoaded();

            lock (_lock)
            {
                return _runtime.Synthesize(
                    text,
                    voice: voice,
                    promptAudioPath: promptAudioPath,
                    outputPath: outputPath,
                    maxNewFrames: maxNewFrames,
                    voiceCloneMaxTextTokens: voiceCloneMaxTextTokens,
                    doSample: doSample,
                    sampleMode: sampleMode,
                    streaming: streaming,
                    seed: seed);
            }
        }

        /// <summary>
        /// 异步合成语音
        /// </summary>
        public Task<SynthesisResult> SynthesizeAsync(
            string text,
            string voice = null,
            string promptAudioPath = null,
            string outputPath = null,
            int? maxNewFrames = null,
            int voiceCloneMaxTextTokens = 75,
            bool doSample = true,
            string sampleMode = "fixed",
            bool streaming = false,
            int? seed = null,
            CancellationToken cancellationToken = default)
        {
            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return Synthesize(
                    text, voice, promptAudioPath, outputPath,
                    maxNewFrames, voiceCloneMaxTextTokens,
                    doSample, sampleMode, streaming, seed);
            }, cancellationToken);
        }

        /// <summary>
        /// 流式合成
        /// </summary>
        public List<AudioChunkEvent> SynthesizeStream(
            string text,
            string voice = null,
            string promptAudioPath = null,
            int? maxNewFrames = null,
            int voiceCloneMaxTextTokens = 75,
            int? seed = null)
        {
            EnsureLoaded();

            lock (_lock)
            {
                return _runtime.SynthesizeStream(
                    text,
                    voice: voice,
                    promptAudioPath: promptAudioPath,
                    maxNewFrames: maxNewFrames,
                    voiceCloneMaxTextTokens: voiceCloneMaxTextTokens,
                    seed: seed);
            }
        }

        /// <summary>
        /// 获取可用语音列表
        /// </summary>
        public List<string> ListVoices()
        {
            EnsureLoaded();
            return _runtime.Manifest.builtin_voices?.Select(v => v.voice).ToList() ?? new List<string>();
        }

        /// <summary>
        /// 文本分块（用于长文本语音克隆）
        /// </summary>
        public List<string> SplitVoiceCloneText(string text, int maxTokens = 75)
        {
            EnsureLoaded();
            return _runtime.SplitVoiceCloneText(text, maxTokens);
        }

        /// <summary>
        /// 编码参考音频
        /// </summary>
        public List<int[]> EncodeReferenceAudio(string audioPath)
        {
            EnsureLoaded();
            return _runtime.EncodeReferenceAudio(audioPath);
        }

        private void EnsureLoaded()
        {
            if (!_isLoaded)
                throw new InvalidOperationException("Model not loaded. Call LoadModel() first.");
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    _runtime?.Dispose();
                    _runtime = null;
                    _isLoaded = false;
                }
                _disposed = true;
            }
        }
    }
}
