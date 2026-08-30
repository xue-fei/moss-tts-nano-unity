using System;
using System.Collections.Generic;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// 推理引擎接口 - 封装 ONNX 模型推理
    /// 可接入 Barracuda、OnnxRuntime 或其他后端
    /// </summary>
    public interface IInferenceEngine : IDisposable
    {
        /// <summary>
        /// 是否有 LocalGreedyFrame 模型
        /// </summary>
        bool HasLocalGreedyFrame { get; }

        /// <summary>
        /// 是否有 LocalFixedSampledFrame 模型
        /// </summary>
        bool HasLocalFixedSampledFrame { get; }

        /// <summary>
        /// 是否有 LocalCachedStep 模型
        /// </summary>
        bool HasLocalCachedStep { get; }

        /// <summary>
        /// 加载模型清单
        /// </summary>
        void LoadManifest(string manifestPath);

        /// <summary>
        /// Prefill 阶段 - 处理输入序列
        /// </summary>
        /// <param name="inputIds">输入 token IDs [1, seqLen, nVq+1]</param>
        /// <param name="attentionMask">注意力掩码 [1, seqLen]</param>
        /// <returns>(globalHidden, pastStates, pastValidLength)</returns>
        (float[] globalHidden, Dictionary<string, float[]> pastStates, int pastValidLength) Prefill(
            int[,,] inputIds, int[,] attentionMask);

        /// <summary>
        /// Decode 阶段 - 单步解码
        /// </summary>
        /// <param name="inputIds">单步输入 [1, 1, nVq+1]</param>
        /// <param name="pastValidLength">当前 past 有效长度</param>
        /// <param name="pastStates">past 状态</param>
        /// <returns>(globalHidden, newPastStates)</returns>
        (float[] globalHidden, Dictionary<string, float[]> newPastStates) Decode(
            int[,,] inputIds, int pastValidLength, Dictionary<string, float[]> pastStates);

        /// <summary>
        /// 本地解码器 - 生成 text/audio logits
        /// </summary>
        /// <param name="globalHidden">全局隐藏状态</param>
        /// <param name="textTokenId">文本 token ID</param>
        /// <param name="audioPrefix">音频前缀</param>
        /// <returns>(textLogits, audioLogits)</returns>
        (float[] textLogits, float[] audioLogits) LocalDecoder(
            float[] globalHidden, int textTokenId, int[] audioPrefix);

        /// <summary>
        /// 本地 Greedy Frame - 生成完整帧
        /// </summary>
        /// <param name="globalHidden">全局隐藏状态</param>
        /// <param name="previousTokenSetsByChannel">每通道已见 token 集合</param>
        /// <param name="repetitionPenalty">重复惩罚</param>
        /// <returns>(shouldContinue, frameTokenIds)</returns>
        (bool shouldContinue, int[] frameTokenIds) LocalGreedyFrame(
            float[] globalHidden, List<HashSet<int>> previousTokenSetsByChannel, float repetitionPenalty);

        /// <summary>
        /// 本地 Fixed Sampled Frame - 采样完整帧
        /// </summary>
        /// <param name="globalHidden">全局隐藏状态</param>
        /// <param name="previousTokenSetsByChannel">每通道已见 token 集合</param>
        /// <param name="rng">随机数生成器</param>
        /// <returns>(shouldContinue, frameTokenIds)</returns>
        (bool shouldContinue, int[] frameTokenIds) LocalFixedSampledFrame(
            float[] globalHidden, List<HashSet<int>> previousTokenSetsByChannel, System.Random rng);

        /// <summary>
        /// 本地 Cached Step - 带缓存的单步
        /// </summary>
        (float[] textLogits, float[] audioLogits, Dictionary<string, float[]> nextPast) LocalCachedStep(
            float[] globalHidden, int textTokenId, int audioTokenId,
            int channelIndex, int stepType, int pastValidLengths,
            Dictionary<string, float[]> localPastByName);

        /// <summary>
        /// 获取音频 logits 中指定通道的切片
        /// </summary>
        float[] SliceAudioChannelLogits(float[] audioLogits, int channelIndex, int nVq, int codebookSize);

        /// <summary>
        /// Codec 编码 - 音频波形转 token codes
        /// </summary>
        List<int[]> CodecEncode(float[] waveform, int sampleRate);

        /// <summary>
        /// Codec 解码 - token codes 转音频波形
        /// </summary>
        (float[][] channelArrays, int audioLength) CodecDecode(List<int[]> generatedFrames);

        /// <summary>
        /// Codec 流式解码 - 增量解码
        /// </summary>
        (float[] audio, int audioLength) CodecDecodeStep(List<int[]> frameRows);

        /// <summary>
        /// 重置流式解码状态
        /// </summary>
        void CodecDecodeStepReset();

        /// <summary>
        /// 初始化所有推理会话（ONNX Runtime 专用）
        /// </summary>
        void InitializeSessions(int threadCount = 4);
    }

    /// <summary>
    /// 推理引擎工厂
    /// </summary>
    public static class InferenceEngineFactory
    {
        public static IInferenceEngine Create(string backend = "onnxruntime")
        {
            switch (backend.ToLower())
            {
                case "onnxruntime":
                    return new OnnxRuntimeEngine();
                case "barracuda":
                    return new BarracudaInferenceEngine();
                default:
                    throw new ArgumentException($"Unknown backend: {backend}");
            }
        }
    }
}
