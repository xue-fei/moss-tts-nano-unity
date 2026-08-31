using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// 数学工具函数 - 对应 Python 的 numpy 操作
    /// </summary>
    public static class MossTtsMath
    {
        /// <summary>
        /// Softmax 函数
        /// </summary>
        public static float[] Softmax(float[] values)
        {
            if (values == null || values.Length == 0) return Array.Empty<float>();

            float max = values.Max();
            float sum = 0f;
            float[] result = new float[values.Length];

            for (int i = 0; i < values.Length; i++)
            {
                result[i] = Mathf.Exp(values[i] - max);
                sum += result[i];
            }

            if (sum > 0)
            {
                for (int i = 0; i < result.Length; i++)
                    result[i] /= sum;
            }

            return result;
        }

        /// <summary>
        /// 从 logits 中采样 token
        /// </summary>
        public static int SampleFromScores(
            float[] values,
            bool doSample,
            float temperature,
            int topK,
            float topP,
            System.Random rng)
        {
        {
            if (!doSample)
                return Argmax(values);

            if (temperature <= 0f)
                throw new ArgumentException("temperature must be positive when doSample=true");

            float[] scores = new float[values.Length];
            for (int i = 0; i < values.Length; i++)
                scores[i] = values[i] / temperature;

            // Top-K 过滤
            if (topK > 0 && topK < scores.Length)
            {
                var sorted = scores.OrderByDescending(v => v).ToArray();
                float threshold = sorted[topK - 1];
                for (int i = 0; i < scores.Length; i++)
                    if (scores[i] < threshold) scores[i] = float.MinValue;
            }

            // Top-P 过滤
            if (topP > 0f && topP < 1f)
            {
                var indexed = scores.Select((v, i) => (index: i, value: v))
                    .OrderByDescending(x => x.value).ToList();
                float[] sortedScores = indexed.Select(x => x.value).ToArray();
                float[] sortedProbs = Softmax(sortedScores);

                bool[] removeMask = new bool[indexed.Count];
                float cumulative = 0f;
                for (int i = 0; i < sortedProbs.Length; i++)
                {
                    cumulative += sortedProbs[i];
                    if (cumulative > topP)
                        removeMask[i] = true;
                }

                for (int i = removeMask.Length - 1; i > 0; i--)
                    removeMask[i] = removeMask[i - 1];
                if (removeMask.Length > 0) removeMask[0] = false;

                for (int i = 0; i < removeMask.Length; i++)
                    if (removeMask[i])
                        scores[indexed[i].index] = float.MinValue;
            }

            float[] probabilities = Softmax(scores);
            float randomValue = (float)rng.NextDouble();

            for (int i = 0; i < probabilities.Length; i++)
            {
                randomValue -= probabilities[i];
                if (randomValue <= 0f)
                    return i;
            }

            return Argmax(scores);
        }

        /// <summary>
        /// 带重复惩罚的 argmax
        /// </summary>
        public static int ArgmaxWithRepetitionPenalty(float[] values, HashSet<int> previousTokenSet, float repetitionPenalty)
        {
            if (previousTokenSet == null || previousTokenSet.Count == 0 || repetitionPenalty == 1.0f)
                return Argmax(values);

            int bestIndex = 0;
            float bestValue = float.MinValue;
            bool applyPenalty = repetitionPenalty != 1.0f;

            for (int i = 0; i < values.Length; i++)
            {
                float score = values[i];
                if (applyPenalty && previousTokenSet.Contains(i))
                    score = score < 0 ? score * repetitionPenalty : score / repetitionPenalty;

                if (score > bestValue)
                {
                    bestValue = score;
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        /// <summary>
        /// 应用重复惩罚
        /// </summary>
        public static float[] ApplyRepetitionPenalty(float[] values, List<int> previousTokenIds, float repetitionPenalty)
        {
            if (previousTokenIds == null || previousTokenIds.Count == 0 || repetitionPenalty == 1.0f)
                return values;

            float[] result = (float[])values.Clone();
            HashSet<int> uniqueIds = new HashSet<int>(previousTokenIds);

            foreach (int tokenId in uniqueIds)
            {
                if (tokenId < 0 || tokenId >= result.Length) continue;
                result[tokenId] = result[tokenId] < 0
                    ? result[tokenId] * repetitionPenalty
                    : result[tokenId] / repetitionPenalty;
            }

            return result;
        }

        /// <summary>
        /// 返回数组中最大值的索引
        /// </summary>
        public static int Argmax(float[] values)
        {
            if (values == null || values.Length == 0) return 0;
            int bestIndex = 0;
            float bestValue = values[0];
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > bestValue)
                {
                    bestValue = values[i];
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// 辅助文本 token 采样（仅从 assistant slot 和 end token 中选择）
        /// </summary>
        public static int SampleAssistantTextToken(
            float[] textLogits,
            int assistantSlotTokenId,
            int endTokenId,
            GenerationConfig config,
            System.Random rng)
        {
            int[] candidateIds = { assistantSlotTokenId, endTokenId };
            float[] candidateScores = { textLogits[assistantSlotTokenId], textLogits[endTokenId] };

            int sampledIndex = SampleFromScores(
                candidateScores,
                config.do_sample,
                config.text_temperature,
                Mathf.Min(config.text_top_k, candidateScores.Length),
                config.text_top_p,
                rng);

            return candidateIds[sampledIndex];
        }

        /// <summary>
        /// 音频 token 采样
        /// </summary>
        public static int SampleAudioToken(
            float[] audioLogits,
            List<int> previousTokenIds,
            HashSet<int> previousTokenSet,
            GenerationConfig config,
            System.Random rng)
        {
            if (!config.do_sample)
                return ArgmaxWithRepetitionPenalty(audioLogits, previousTokenSet, config.audio_repetition_penalty);

            float[] penalizedScores = ApplyRepetitionPenalty(audioLogits, previousTokenIds, config.audio_repetition_penalty);
            return SampleFromScores(
                penalizedScores,
                true,
                config.audio_temperature,
                config.audio_top_k,
                config.audio_top_p,
                rng);
        }
    }
}
