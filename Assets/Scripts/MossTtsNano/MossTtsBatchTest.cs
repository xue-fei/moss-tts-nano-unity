using System;
using System.IO;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// MOSS-TTS-Nano 批量测试脚本
    /// 在 Unity Editor 中通过菜单项 "Tools/MOSS-TTS-Nano/Run Batch Test" 运行
    /// 或在命令行中用 -batchmode -executeMethod MossTtsBatchTest.Run 运行
    /// </summary>
    public static class MossTtsBatchTest
    {
        public static void Run()
        {
            Debug.Log("[MossTtsBatchTest] Starting batch test...");

            // 模型位于 StreamingAssets/Models 下（.onnx/.data 不是 Unity 资源类型，
            // 放在 Assets 里打包时不会被复制到运行时目录）
            string modelDir = MossTtsComponent.ResolveModelDir();
            string outputDir = Path.Combine(Application.persistentDataPath, "MossTtsOutput");
            Directory.CreateDirectory(outputDir);

            // 检查模型文件
            if (!Directory.Exists(modelDir))
            {
                Debug.LogError($"[MossTtsBatchTest] Model directory not found: {modelDir}");
                return;
            }

            string manifestPath = Path.Combine(modelDir, "browser_poc_manifest.json");
            if (!File.Exists(manifestPath))
            {
                Debug.LogError($"[MossTtsBatchTest] Manifest not found: {manifestPath}");
                return;
            }

            Debug.Log($"[MossTtsBatchTest] Model directory: {modelDir}");
            Debug.Log($"[MossTtsBatchTest] Output directory: {outputDir}");

            try
            {
                // 创建服务
                var service = new MossTtsService(modelDir, outputDir, 4, "cpu");
                service.LoadModel();
                Debug.Log("[MossTtsBatchTest] Model loaded successfully!");

                // 获取语音列表
                var voices = service.ListVoices();
                Debug.Log($"[MossTtsBatchTest] Available voices: {string.Join(", ", voices)}");

                // 测试合成
                string[] testTexts = {
                    "欢迎关注模思智能。",
                    "这是一个语音合成测试。",
                    "MOSS TTS Nano 是一款轻量级语音合成模型。"
                };

                for (int i = 0; i < testTexts.Length; i++)
                {
                    string text = testTexts[i];
                    Debug.Log($"[MossTtsBatchTest] Test {i + 1}/{testTexts.Length}: {text}");

                    var result = service.Synthesize(
                        text,
                        voice: "Junhao",
                        maxNewFrames: 96,
                        doSample: false);

                    Debug.Log($"[MossTtsBatchTest] Result: {result.Waveform.Length} samples, " +
                             $"{result.Waveform.Length / (float)result.SampleRate:F2}s, " +
                             $"saved to {result.AudioPath}");
                }

                Debug.Log("[MossTtsBatchTest] All tests completed!");
                service.Dispose();
            }
            catch (Exception e)
            {
                Debug.LogError($"[MossTtsBatchTest] Error: {e.Message}\n{e.StackTrace}");
            }
        }
    }
}
