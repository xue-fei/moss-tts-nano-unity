using System;
using System.Collections;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;

namespace MossTtsNano
{
    /// <summary>
    /// MOSS-TTS-Nano Unity 组件 - 挂载到 GameObject 即可使用
    /// 支持：
    /// - 配置模型路径、输出路径
    /// - 合成语音并输出 AudioClip
    /// - 流式合成实时播放
    /// - 语音克隆（通过参考音频）
    /// </summary>
    public class MossTtsComponent : MonoBehaviour
    {
        [Header("模型配置")]
        [Tooltip("模型目录路径（包含 browser_poc_manifest.json）")]
        public string ModelDir = "Models/MOSS-TTS-Nano-ONNX";

        [Tooltip("输出目录")]
        public string OutputDir = "";

        [Tooltip("推理线程数")]
        public int ThreadCount = 4;

        [Tooltip("执行提供者 (cpu/cuda)")]
        public string ExecutionProvider = "cpu";

        [Header("合成参数")]
        [Tooltip("默认语音")]
        public string DefaultVoice = "Junhao";

        [Tooltip("最大生成帧数")]
        public int MaxNewFrames = 375;

        [Tooltip("语音克隆最大 token 数")]
        public int VoiceCloneMaxTextTokens = 75;

        [Tooltip("是否采样")]
        public bool DoSample = true;

        [Tooltip("采样模式 (greedy/fixed/full)")]
        public string SampleMode = "fixed";

        [Tooltip("文本温度")]
        public float TextTemperature = 1.0f;

        [Tooltip("音频温度")]
        public float AudioTemperature = 0.8f;

        [Tooltip("音频重复惩罚")]
        public float AudioRepetitionPenalty = 1.2f;

        [Header("流式合成")]
        [Tooltip("启用流式合成")]
        public bool Streaming = false;

        [Tooltip("流式合成播放延迟（秒）")]
        public float StreamPlayDelay = 0.1f;

        [Header("事件")]
        public UnityEvent<SynthesisResult> OnSynthesisComplete;
        public UnityEvent<AudioChunkEvent> OnAudioChunk;
        public UnityEvent<string> OnError;
        public UnityEvent OnModelLoaded;

        private MossTtsService _service;
        private ConcurrentQueue<Action> _mainThreadQueue;
        private CancellationTokenSource _streamCts;
        private AudioSource _audioSource;
        private bool _isStreaming;

        /// <summary>
        /// 是否已加载模型
        /// </summary>
        public bool IsModelLoaded => _service?.IsLoaded ?? false;

        private void Awake()
        {
            _mainThreadQueue = new ConcurrentQueue<Action>();
            _audioSource = GetComponent<AudioSource>();
            if (_audioSource == null && Streaming)
                _audioSource = gameObject.AddComponent<AudioSource>();
        }

        private void Start()
        {
            LoadModel();
        }

        private void Update()
        {
            // 主线程回调
            while (_mainThreadQueue.TryDequeue(out var action))
            {
                try
                {
                    action?.Invoke();
                }
                catch (Exception e)
                {
                    Debug.LogError($"[MossTts] Main thread callback error: {e.Message}");
                }
            }
        }

        private void OnDestroy()
        {
            _streamCts?.Cancel();
            _service?.Dispose();
        }

        /// <summary>
        /// 加载模型
        /// </summary>
        public void LoadModel()
        {
            try
            {
                // 尝试多个可能路径
                string[] candidatePaths = {
                    Path.Combine(Application.streamingAssetsPath, ModelDir),
                    Path.Combine(Application.dataPath, ModelDir),
                    Path.Combine("Assets", ModelDir),
                    ModelDir
                };

                string modelDir = null;
                foreach (var path in candidatePaths)
                {
                    string fullPath = Path.GetFullPath(path);
                    if (Directory.Exists(fullPath) && File.Exists(Path.Combine(fullPath, "browser_poc_manifest.json")))
                    {
                        modelDir = fullPath;
                        break;
                    }
                }

                if (modelDir == null)
                {
                    Debug.LogError($"[MossTts] Model directory not found. Searched: {string.Join(", ", candidatePaths)}");
                    OnError?.Invoke("Model directory not found");
                    return;
                }

                string outputDir = string.IsNullOrEmpty(OutputDir)
                    ? Path.Combine(Application.persistentDataPath, "MossTtsOutput")
                    : OutputDir;

                _service = new MossTtsService(modelDir, outputDir, ThreadCount, ExecutionProvider);
                _service.LoadModel();
                OnModelLoaded?.Invoke();
                Debug.Log($"[MossTts] Model loaded from {modelDir}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[MossTts] Failed to load model: {e.Message}\n{e.StackTrace}");
                OnError?.Invoke($"Failed to load model: {e.Message}");
            }
        }

        /// <summary>
        /// 合成语音
        /// </summary>
        public void Synthesize(
            string text,
            string voice = null,
            string promptAudioPath = null,
            string outputPath = null)
        {
            if (_service == null || !_service.IsLoaded)
            {
                OnError?.Invoke("Model not loaded");
                return;
            }

            _ = SynthesizeInternalAsync(text, voice, promptAudioPath, outputPath);
        }

        private async Task SynthesizeInternalAsync(
            string text, string voice, string promptAudioPath, string outputPath)
        {
            try
            {
                var result = await _service.SynthesizeAsync(
                    text,
                    voice: voice ?? DefaultVoice,
                    promptAudioPath: promptAudioPath,
                    outputPath: outputPath,
                    maxNewFrames: MaxNewFrames,
                    voiceCloneMaxTextTokens: VoiceCloneMaxTextTokens,
                    doSample: DoSample,
                    sampleMode: SampleMode);

                _mainThreadQueue.Enqueue(() =>
                {
                    OnSynthesisComplete?.Invoke(result);
                    Debug.Log($"[MossTts] Synthesis complete: {result.AudioPath} ({result.Waveform.Length} samples)");
                });
            }
            catch (Exception e)
            {
                _mainThreadQueue.Enqueue(() =>
                {
                    OnError?.Invoke($"Synthesis failed: {e.Message}");
                });
            }
        }

        /// <summary>
        /// 合成并播放
        /// </summary>
        public void SynthesizeAndPlay(string text, string voice = null, string promptAudioPath = null)
        {
            if (_service == null || !_service.IsLoaded)
            {
                OnError?.Invoke("Model not loaded");
                return;
            }

            _ = SynthesizeAndPlayInternalAsync(text, voice, promptAudioPath);
        }

        private async Task SynthesizeAndPlayInternalAsync(string text, string voice, string promptAudioPath)
        {
            try
            {
                var result = await _service.SynthesizeAsync(
                    text,
                    voice: voice ?? DefaultVoice,
                    promptAudioPath: promptAudioPath,
                    maxNewFrames: MaxNewFrames,
                    voiceCloneMaxTextTokens: VoiceCloneMaxTextTokens,
                    doSample: DoSample,
                    sampleMode: SampleMode);

                _mainThreadQueue.Enqueue(() =>
                {
                    PlayWaveform(result.Waveform, result.SampleRate, result.Channels);
                    OnSynthesisComplete?.Invoke(result);
                });
            }
            catch (Exception e)
            {
                _mainThreadQueue.Enqueue(() =>
                {
                    OnError?.Invoke($"Synthesis failed: {e.Message}");
                });
            }
        }

        /// <summary>
        /// 流式合成并播放
        /// </summary>
        public void SynthesizeStreamAndPlay(string text, string voice = null, string promptAudioPath = null)
        {
            if (_service == null || !_service.IsLoaded)
            {
                OnError?.Invoke("Model not loaded");
                return;
            }

            if (_isStreaming)
            {
                Debug.LogWarning("[MossTts] Already streaming, stopping previous");
                StopStream();
            }

            _streamCts = new CancellationTokenSource();
            _ = SynthesizeStreamInternalAsync(text, voice, promptAudioPath, _streamCts.Token);
        }

        private async Task SynthesizeStreamInternalAsync(
            string text, string voice, string promptAudioPath, CancellationToken ct)
        {
            _isStreaming = true;

            try
            {
                var chunks = await Task.Run(() =>
                {
                    return _service.SynthesizeStream(
                        text,
                        voice: voice ?? DefaultVoice,
                        promptAudioPath: promptAudioPath,
                        maxNewFrames: MaxNewFrames,
                        voiceCloneMaxTextTokens: VoiceCloneMaxTextTokens);
                }, ct);

                _mainThreadQueue.Enqueue(() =>
                {
                    foreach (var chunk in chunks)
                    {
                        OnAudioChunk?.Invoke(chunk);
                    }
                });
            }
            catch (OperationCanceledException)
            {
                Debug.Log("[MossTts] Stream synthesis cancelled");
            }
            catch (Exception e)
            {
                _mainThreadQueue.Enqueue(() =>
                {
                    OnError?.Invoke($"Stream synthesis failed: {e.Message}");
                });
            }
            finally
            {
                _isStreaming = false;
            }
        }

        /// <summary>
        /// 停止流式合成
        /// </summary>
        public void StopStream()
        {
            _streamCts?.Cancel();
            _isStreaming = false;
            if (_audioSource != null && _audioSource.isPlaying)
                _audioSource.Stop();
        }

        /// <summary>
        /// 播放波形数据
        /// </summary>
        public void PlayWaveform(float[] waveform, int sampleRate, int channels)
        {
            if (_audioSource == null)
            {
                _audioSource = gameObject.AddComponent<AudioSource>();
            }

            AudioClip clip = AudioClip.Create("MossTtsAudio", waveform.Length / channels, channels, sampleRate, false);
            // 转换为 float[] 并设置数据
            float[] monowave = new float[waveform.Length / Mathf.Max(1, channels)];
            for (int i = 0; i < monowave.Length; i++)
            {
                // 取所有通道的平均值
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                    sum += waveform[i * channels + c];
                monowave[i] = sum / channels;
            }
            clip.SetData(monowave, 0);

            _audioSource.clip = clip;
            _audioSource.Play();
        }

        /// <summary>
        /// 获取可用语音列表
        /// </summary>
        public string[] GetVoiceList()
        {
            if (_service == null || !_service.IsLoaded)
                return Array.Empty<string>();

            return _service.ListVoices().ToArray();
        }

        /// <summary>
        /// 文本分块预览
        /// </summary>
        public string[] PreviewTextChunks(string text, int maxTokens = 75)
        {
            if (_service == null || !_service.IsLoaded)
                return new[] { text };

            return _service.SplitVoiceCloneText(text, maxTokens).ToArray();
        }
    }
}
