using System;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

namespace MossTtsNano
{
    /// <summary>
    /// MOSS-TTS-Nano 测试脚本 - 提供简单的 UI 测试界面
    /// 挂载到带有 MossTtsComponent 的 GameObject 上即可
    /// </summary>
    public class MossTtsTest : MonoBehaviour
    {
        [Header("UI References")]
        public InputField textInput;
        public Button synthesizeButton;
        public Button synthesizeAndPlayButton;
        public Button streamButton;
        public Text statusText;
        public Text resultText;
        public Dropdown voiceDropdown;

        [Header("Test Settings")]
        public string testText = "欢迎关注模思智能，这是一个语音合成测试。";
        public string modelPath = "Assets/Models/MOSS-TTS-Nano-ONNX";

        private MossTtsComponent _ttsComponent;
        private string _outputDir;

        void Start()
        {
            _ttsComponent = GetComponent<MossTtsComponent>();
            _outputDir = Path.Combine(Application.persistentDataPath, "MossTtsOutput");
            Directory.CreateDirectory(_outputDir);

            if (_ttsComponent == null)
            {
                Debug.LogError("[MossTtsTest] MossTtsComponent not found on this GameObject!");
                UpdateStatus("Error: MossTtsComponent not found");
                return;
            }

            SetupUI();
            UpdateStatus("Ready. Click a button to test.");
        }

        void SetupUI()
        {
            if (synthesizeButton != null)
                synthesizeButton.onClick.AddListener(OnSynthesizeClicked);
            if (synthesizeAndPlayButton != null)
                synthesizeAndPlayButton.onClick.AddListener(OnSynthesizeAndPlayClicked);
            if (streamButton != null)
                streamButton.onClick.AddListener(OnStreamClicked);

            // 绑定事件
            _ttsComponent.OnModelLoaded.AddListener(OnModelLoaded);
            _ttsComponent.OnSynthesisComplete.AddListener(OnSynthesisComplete);
            _ttsComponent.OnError.AddListener(OnError);
            _ttsComponent.OnAudioChunk.AddListener(OnAudioChunk);
        }

        void OnDestroy()
        {
            if (_ttsComponent != null)
            {
                _ttsComponent.OnModelLoaded.RemoveListener(OnModelLoaded);
                _ttsComponent.OnSynthesisComplete.RemoveListener(OnSynthesisComplete);
                _ttsComponent.OnError.RemoveListener(OnError);
                _ttsComponent.OnAudioChunk.RemoveListener(OnAudioChunk);
            }
        }

        void Update()
        {
            // 空格键快速测试
            if (Input.GetKeyDown(KeyCode.Space))
            {
                OnSynthesizeAndPlayClicked();
            }
            // ESC 停止流式
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                _ttsComponent.StopStream();
                UpdateStatus("Stream stopped");
            }
        }

        public void OnSynthesizeClicked()
        {
            string text = GetInputText();
            if (string.IsNullOrEmpty(text))
            {
                UpdateStatus("Error: Please enter text");
                return;
            }

            UpdateStatus("Synthesizing...");
            _ttsComponent.Synthesize(text);
        }

        public void OnSynthesizeAndPlayClicked()
        {
            string text = GetInputText();
            if (string.IsNullOrEmpty(text))
            {
                UpdateStatus("Error: Please enter text");
                return;
            }

            UpdateStatus("Synthesizing and playing...");
            _ttsComponent.SynthesizeAndPlay(text);
        }

        public void OnStreamClicked()
        {
            string text = GetInputText();
            if (string.IsNullOrEmpty(text))
            {
                UpdateStatus("Error: Please enter text");
                return;
            }

            UpdateStatus("Streaming...");
            _ttsComponent.Streaming = true;
            _ttsComponent.SynthesizeStreamAndPlay(text);
        }

        void OnModelLoaded()
        {
            UpdateStatus("Model loaded successfully!");
        }

        void OnSynthesisComplete(SynthesisResult result)
        {
            string info = $"Synthesis complete!\n" +
                         $"Samples: {result.Waveform.Length}\n" +
                         $"SampleRate: {result.SampleRate}\n" +
                         $"Channels: {result.Channels}\n" +
                         $"Duration: {result.Waveform.Length / (float)result.SampleRate:F2}s\n" +
                         $"File: {result.AudioPath}\n" +
                         $"Chunks: {result.TextChunks?.Length ?? 0}";
            UpdateStatus(info);
        }

        void OnError(string error)
        {
            UpdateStatus($"Error: {error}");
            Debug.LogError($"[MossTtsTest] {error}");
        }

        void OnAudioChunk(AudioChunkEvent chunk)
        {
            UpdateStatus($"Audio chunk {chunk.ChunkIndex}: {chunk.Waveform.Length} samples, " +
                        $"emitted: {chunk.EmittedAudioSeconds:F2}s");
        }

        string GetInputText()
        {
            if (textInput != null && !string.IsNullOrEmpty(textInput.text))
                return textInput.text;
            return testText;
        }

        void UpdateStatus(string message)
        {
            Debug.Log($"[MossTtsTest] {message}");
            if (statusText != null)
                statusText.text = message;
        }
    }
}
