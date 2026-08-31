using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// 自动设置测试场景 - 创建必要的 GameObject 和组件
    /// 在空场景中添加此脚本即可快速开始测试
    /// </summary>
    public class MossTtsSceneSetup : MonoBehaviour
    {
        [Header("Scene Setup")]
        public bool createTTSObject = true;
        public bool createSimpleTest = true;
        public bool createCanvas = false;

        [Tooltip("模型目录，相对于 StreamingAssets")]
        public string modelPath = MossTtsComponent.DefaultModelDir;

        [Header("Test Text")]
        public string testText = "欢迎关注模思智能，这是一个语音合成测试。";

        void Awake()
        {
            if (createTTSObject)
            {
                CreateTTSObject();
            }
        }

        void CreateTTSObject()
        {
            // 创建 TTS GameObject
            GameObject ttsObj = new GameObject("MossTTS");
            ttsObj.transform.SetParent(transform);

            // 添加 MossTtsComponent
            // modelPath 走统一归一化，兼容 Inspector 里残留的 "Assets/Models/..." 旧值
            string relativeModelDir = MossTtsComponent.NormalizeModelDir(modelPath);
            var ttsComponent = ttsObj.AddComponent<MossTtsComponent>();
            ttsComponent.ModelDir = relativeModelDir;

            // 添加 AudioSource
            var audioSource = ttsObj.AddComponent<UnityEngine.AudioSource>();
            audioSource.playOnAwake = false;

            // 添加测试脚本
            if (createSimpleTest)
            {
                var test = ttsObj.AddComponent<MossTtsSimpleTest>();
                // 传归一化后的值，避免 Inspector 上显示已失效的旧路径
                test.modelPath = relativeModelDir;
                test.testText = testText;
            }

            Debug.Log("[MossTtsSceneSetup] MossTTS object created. Press Play to start.");
        }
    }
}
