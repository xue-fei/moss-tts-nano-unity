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
        /// <summary>
        /// 模型目录相对 StreamingAssets 的默认位置。
        /// 模型已从 Assets/Models 移到 Assets/StreamingAssets/Models，
        /// 因为 .onnx / .data 不是 Unity 可识别的资源类型，放在 Assets 下
        /// 打包时不会被复制到运行时目录；StreamingAssets 才会原样输出。
        /// </summary>
        public const string DefaultModelDir = "Models/MOSS-TTS-Nano-ONNX";

        /// <summary>
        /// 模型目录的绝对路径。运行时与 Editor 下都指向 StreamingAssets。
        /// </summary>
        public static string ResolveModelDir()
        {
            return Path.Combine(Application.streamingAssetsPath, DefaultModelDir);
        }

        /// <summary>
        /// 把历史遗留的路径写法归一化为「相对 StreamingAssets」的形式。
        /// 处理 "Assets/StreamingAssets/Models/X"、"Assets/Models/X"、
        /// "StreamingAssets/Models/X" 三种旧值，全部收敛到 "Models/X"。
        /// </summary>
        public static string NormalizeModelDir(string dir)
        {
            if (string.IsNullOrEmpty(dir)) return DefaultModelDir;

            string normalized = dir.Replace('\\', '/').TrimStart('/');

            if (normalized.StartsWith("Assets/", StringComparison.OrdinalIgnoreCase))
                normalized = normalized.Substring("Assets/".Length);

            if (normalized.StartsWith("StreamingAssets/", StringComparison.OrdinalIgnoreCase))
                normalized = normalized.Substring("StreamingAssets/".Length);

            return normalized;
        }

        [Header("模型配置")]
        [Tooltip("模型目录，相对于 StreamingAssets（包含 browser_poc_manifest.json）")]
        public string ModelDir = DefaultModelDir;

        [Tooltip("输出目录")]
        public string OutputDir = "";

        [Tooltip("推理线程数")]
        public int ThreadCount = 4;

        [Tooltip("执行提供者 (cpu/cuda)")]
        public string ExecutionProvider = "cuda";

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

        [Tooltip("随机种子。留空(-1)表示每次随机；填正整数可复现同一条音频。\n" +
                 "模型在部分 seed 下会自己生成大段静音，固定一个听感好的 seed 是最省事的规避手段。")]
        public int Seed = -1;

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
                string relativeDir = NormalizeModelDir(ModelDir);
                string modelDir = Path.GetFullPath(
                    Path.Combine(Application.streamingAssetsPath, relativeDir));

                if (!Directory.Exists(modelDir) ||
                    !File.Exists(Path.Combine(modelDir, "browser_poc_manifest.json")))
                {
                    Debug.LogError(
                        $"[MossTts] Model directory not found or missing browser_poc_manifest.json: {modelDir}\n" +
                        $"(ModelDir='{ModelDir}' → '{relativeDir}', " +
                        $"streamingAssetsPath='{Application.streamingAssetsPath}')");
                    OnError?.Invoke("Model directory not found");
                    return;
                }

                string outputDir = string.IsNullOrEmpty(OutputDir)
                    ? Path.Combine(Application.dataPath, "MossTtsOutput")
                    : OutputDir;

                _service = new MossTtsService(modelDir, outputDir, ThreadCount, ExecutionProvider);
                _service.LoadModel();
                OnModelLoaded?.Invoke();

                // 明确区分"请求的 EP"和"实际生效的 EP"。CUDA 注册失败会静默回退到 CPU，
                // 只看 Inspector 上的 ExecutionProvider 字段会误以为在用 GPU。
                string activeEp = _service.ActiveExecutionProvider;
                if (!string.Equals(activeEp, ExecutionProvider, StringComparison.OrdinalIgnoreCase))
                {
                    Debug.LogWarning(
                        $"[MossTts] Requested EP '{ExecutionProvider}' but running on '{activeEp}'. " +
                        "CUDA 需要 onnxruntime-cuda 包的原生库 + 匹配的 CUDA Toolkit/cuDNN 在 PATH 里。");
                }
                Debug.Log($"[MossTts] Model loaded from {modelDir} (EP={activeEp})");
            }
            catch (Exception e)
            {
                Debug.LogError($"[MossTts] Failed to load model: {e.Message}\n{e.StackTrace}");
                OnError?.Invoke($"Failed to load model: {e.Message}");
            }
        }

        /// <summary>
        /// 把 Inspector 上的 Seed 字段翻译成运行时参数。
        /// 负值表示"不固定"，交给运行时用默认时间种子；非负值原样传入以便复现。
        /// </summary>
        private int? ResolveSeed()
        {
            return Seed < 0 ? (int?)null : Seed;
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
                    sampleMode: SampleMode,
                    seed: ResolveSeed());

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
                    sampleMode: SampleMode,
                    seed: ResolveSeed());

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
                        voiceCloneMaxTextTokens: VoiceCloneMaxTextTokens,
                        seed: ResolveSeed());
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
            if (waveform == null || waveform.Length == 0)
            {
                Debug.LogWarning("[MossTts] PlayWaveform called with empty waveform");
                return;
            }

            if (_audioSource == null)
            {
                _audioSource = gameObject.AddComponent<AudioSource>();
            }

            channels = Mathf.Max(1, channels);
            int samplesPerChannel = waveform.Length / channels;
            if (samplesPerChannel <= 0)
            {
                Debug.LogWarning($"[MossTts] Waveform too short: {waveform.Length} values for {channels} channels");
                return;
            }

            // AudioClip.SetData 要求交错排列且长度为 samplesPerChannel * channels，
            // 因此直接写入原始交错数据，不做下混。
            AudioClip clip = AudioClip.Create("MossTtsAudio", samplesPerChannel, channels, sampleRate, false);
            int usable = samplesPerChannel * channels;
            if (usable == waveform.Length)
            {
                clip.SetData(waveform, 0);
            }
            else
            {
                float[] trimmed = new float[usable];
                Array.Copy(waveform, trimmed, usable);
                clip.SetData(trimmed, 0);
            }

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
