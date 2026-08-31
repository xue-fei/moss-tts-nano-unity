using System.IO;

namespace MossTtsNano
{
    /// <summary>
    /// 模型文件的路径解析。
    ///
    /// 背景：模型目录已从 Assets/Models 移到 Assets/StreamingAssets/Models，
    /// 且 codec 目录是 **嵌套** 在 TTS 目录内的：
    ///
    ///   StreamingAssets/Models/MOSS-TTS-Nano-ONNX/
    ///     browser_poc_manifest.json
    ///     tts_browser_onnx_meta.json
    ///     MOSS-Audio-Tokenizer-Nano-ONNX/
    ///       codec_browser_onnx_meta.json
    ///
    /// 而 manifest 里声明的是 codec_meta =
    /// "../MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json"，
    /// 即按「codec 与 TTS 同级」的旧布局写的，在当前布局下解析不到文件。
    /// manifest 是模型发布产物、不方便改，所以由代码兼容两种布局。
    ///
    /// 这份逻辑原先在 OrtCpuRuntime.LoadCodecMeta() 和
    /// OnnxRuntimeEngine.LoadManifest() 各写了一遍（且都只有 fallback 兜着），
    /// 这里收敛成单一实现。
    /// </summary>
    internal static class ModelPaths
    {
        /// <summary>codec meta 的文件名，manifest 路径失效时按目录名拼回来用。</summary>
        private const string CodecMetaFileName = "codec_browser_onnx_meta.json";

        /// <summary>
        /// 解析 manifest 中声明的 codec_meta 路径。
        ///
        /// 按「嵌套布局 → manifest 声明路径 → 同级布局」依次尝试，
        /// 返回第一个真实存在的文件。全部落空时返回嵌套布局的路径，
        /// 让调用方的 File.NotFound 报错指向最可能正确的位置。
        /// </summary>
        /// <param name="modelDir">manifest 所在目录（TTS 模型目录）。</param>
        /// <param name="declaredRelativePath">manifest.model_files.codec_meta 的原值。</param>
        public static string ResolveCodecMeta(string modelDir, string declaredRelativePath)
        {
            // 从声明值里取出 codec 目录名（如 "MOSS-Audio-Tokenizer-Nano-ONNX"），
            // 这样换代模型改名也不用改代码。
            string codecDirName = ExtractCodecDirName(declaredRelativePath);

            // 1) 当前实际布局：codec 嵌套在模型目录内
            string nested = Path.Combine(modelDir, codecDirName, CodecMetaFileName);
            if (File.Exists(nested)) return nested;

            // 2) manifest 声明的原始路径（含 ".."，即 codec 与模型目录同级）
            if (!string.IsNullOrEmpty(declaredRelativePath))
            {
                string declared = Path.GetFullPath(Path.Combine(modelDir, declaredRelativePath));
                if (File.Exists(declared)) return declared;
            }

            // 3) 同级布局但目录名与声明值不同的情况（用 modelDir 的父目录拼）
            string parent = Path.GetDirectoryName(modelDir);
            if (!string.IsNullOrEmpty(parent))
            {
                string sibling = Path.Combine(parent, codecDirName, CodecMetaFileName);
                if (File.Exists(sibling)) return sibling;
            }

            return nested;
        }

        /// <summary>
        /// 从 "../MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json"
        /// 这类相对路径里取出目录名，剥掉前导的 "./" 与 "../"。
        /// </summary>
        private static string ExtractCodecDirName(string declaredRelativePath)
        {
            const string fallbackDirName = "MOSS-Audio-Tokenizer-Nano-ONNX";

            if (string.IsNullOrEmpty(declaredRelativePath)) return fallbackDirName;

            string[] parts = declaredRelativePath.Replace('\\', '/').Split('/');
            for (int i = parts.Length - 2; i >= 0; i--)
            {
                string part = parts[i];
                if (part == "." || part == ".." || part.Length == 0) continue;
                return part;
            }

            return fallbackDirName;
        }
    }
}
