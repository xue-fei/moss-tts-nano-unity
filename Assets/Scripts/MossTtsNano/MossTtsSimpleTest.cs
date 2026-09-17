using System;
using System.IO;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// MOSS-TTS-Nano 简单测试脚本
    /// 挂载到 GameObject 上即可运行，无需 UI
    /// 按 Play 后自动执行测试
    /// </summary>
    public class MossTtsSimpleTest : MonoBehaviour
    {
        [Header("Test Settings")]
        [Tooltip("模型目录，相对于 StreamingAssets（仅作展示，实际由 MossTtsComponent.ModelDir 决定）")]
        public string modelPath = MossTtsComponent.DefaultModelDir;
        public string testText = "欢迎关注模思智能，这是一个语音合成测试。";
        public bool runOnStart = true;
        public bool playAfterSynthesis = true;

        private MossTtsComponent _tts;
        private bool _tested = false;

        void Start()
        {
            _tts = GetComponent<MossTtsComponent>();
            if (_tts == null)
            {
                Debug.LogError("[MossTtsSimpleTest] MossTtsComponent not found!");
                return;
            }

            _tts.OnModelLoaded.AddListener(OnModelLoaded);
            _tts.OnSynthesisComplete.AddListener(OnSynthesisComplete);
            _tts.OnError.AddListener(OnError);

            if (runOnStart)
            {
                // 等待模型加载完成
                if (_tts.IsModelLoaded)
                {
                    RunTest();
                }
                else
                {
                    Debug.Log("[MossTtsSimpleTest] Waiting for model to load...");
                }
            }
        }

        void OnDestroy()
        {
            if (_tts != null)
            {
                _tts.OnModelLoaded.RemoveListener(OnModelLoaded);
                _tts.OnSynthesisComplete.RemoveListener(OnSynthesisComplete);
                _tts.OnError.RemoveListener(OnError);
            }
        }

        void Update()
        {
            // 按 T 键触发测试
            if (Input.GetKeyDown(KeyCode.T) && _tts != null && _tts.IsModelLoaded)
            {
                RunTest();
            }
        }

        void OnModelLoaded()
        {
            Debug.Log("[MossTtsSimpleTest] Model loaded, starting test...");
            RunTest();
        }

        void RunTest()
        {
            if (_tested) return;
            _tested = true;

            Debug.Log($"[MossTtsSimpleTest] Starting synthesis test...");
            Debug.Log($"[MossTtsSimpleTest] Text: {testText}");

            if (playAfterSynthesis)
                _tts.SynthesizeAndPlay(testText);
            else
                _tts.Synthesize(testText);
        }

        void OnSynthesisComplete(SynthesisResult result)
        {
            float duration = result.Waveform.Length / (float)result.SampleRate;
            Debug.Log($"[MossTtsSimpleTest] Synthesis complete!");
            Debug.Log($"[MossTtsSimpleTest] Samples: {result.Waveform.Length}");
            Debug.Log($"[MossTtsSimpleTest] SampleRate: {result.SampleRate}");
            Debug.Log($"[MossTtsSimpleTest] Channels: {result.Channels}");
            Debug.Log($"[MossTtsSimpleTest] Duration: {duration:F2}s");
            Debug.Log($"[MossTtsSimpleTest] Elapsed: {result.ElapsedSeconds:F2}s");
            Debug.Log($"[MossTtsSimpleTest] RTF: {result.RTF:F3}");
            Debug.Log($"[MossTtsSimpleTest] File: {result.AudioPath}");
            Debug.Log($"[MossTtsSimpleTest] Chunks: {result.TextChunks?.Length ?? 0}");

            // 打印前 10 个采样值
            string samples = "First 10 samples: ";
            for (int i = 0; i < Mathf.Min(10, result.Waveform.Length); i++)
                samples += result.Waveform[i].ToString("F4") + " ";
            Debug.Log($"[MossTtsSimpleTest] {samples}");
        }

        void OnError(string error)
        {
            Debug.LogError($"[MossTtsSimpleTest] Error: {error}");
        }
    }
}
