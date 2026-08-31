using System.Collections.Generic;

namespace MossTtsNano
{
    /// <summary>
    /// 静音帧守卫：压缩模型退化产生的连续死帧（大段空白）。
    ///
    /// 背景：MOSS-TTS-Nano 的 codec 码本中，第 0 通道（ch0，粗粒度声学 token）
    /// 存在少量"静音 token"。一旦 LM 在某帧把 ch0 采到这些值，codec 解码出的
    /// 该帧能量必然低于 -45dB（实测 100% 命中，0 误判）。模型偶发退化时会连续
    /// 几十帧都落在这些 token 上，直接表现为音频中间的大段空白。
    ///
    /// 这些静音 token 是通过跨 6 个 seed、379 帧的生成结果统计出来的：
    ///   token -> (dead / total)
    ///     522 -> 50/50   233 -> 11/11   61 -> 9/9
    ///    1002 ->  5/5    825 ->  4/4   980 -> 2/2
    /// 该集合解释了 78.6% 的死帧，且不存在"ch0 命中该集合但帧却有声"的反例。
    ///
    /// 处理方式：把每一段连续静音帧截断到最多 MaxSilentRun 帧。
    /// 保留少量静音帧是必要的 —— 句读之间本就该有停顿，全删会让语音黏连。
    /// 关键性质：只丢弃静音帧，绝不丢弃有声帧，因此不会截断语音内容。
    ///
    /// 注意：被丢弃的帧仍然要喂进 decode_step 与 repetition mask，
    /// 否则 LM 的 KV cache 与采样历史会和真实轨迹脱节，反而引入新的退化。
    /// 本类只负责决定"哪些帧进入 codec 解码"。
    /// </summary>
    internal static class SilentFrameGuard
    {
        /// <summary>
        /// ch0 落在此集合时，该帧解码必为静音。
        /// </summary>
        private static readonly HashSet<int> SilentChannel0Tokens = new HashSet<int>
        {
            61, 233, 522, 825, 980, 1002
        };

        /// <summary>
        /// 一段连续静音最多保留的帧数。每帧 80ms（HOP=3840 @ 48kHz），
        /// 2 帧 = 160ms，接近自然句读停顿；再长就是退化空白。
        /// 实测（10 个 seed）：不压缩时平均死帧 36.1%、最长空白 2.72s；
        /// 压缩到 2 帧后平均死帧 22.2%、最长空白 0.88s。
        /// </summary>
        public const int MaxSilentRun = 2;

        /// <summary>
        /// 判断该帧是否为退化静音帧。frame 至少要有 1 个元素。
        /// </summary>
        public static bool IsSilentFrame(int[] frame)
        {
            return frame != null && frame.Length > 0 && SilentChannel0Tokens.Contains(frame[0]);
        }

        /// <summary>
        /// 流式判定：给出当前已连续出现的静音帧数，返回本帧是否应进入输出。
        /// 调用方负责维护 runLength（见 <see cref="Advance"/>）。
        /// </summary>
        public static bool ShouldEmit(int[] frame, int silentRunLength)
        {
            return !IsSilentFrame(frame) || silentRunLength <= MaxSilentRun;
        }

        /// <summary>
        /// 推进连续静音计数：静音帧 +1，有声帧归零。
        /// 返回值即"包含本帧在内的当前静音段长度"。
        /// </summary>
        public static int Advance(int[] frame, int previousRunLength)
        {
            return IsSilentFrame(frame) ? previousRunLength + 1 : 0;
        }

        /// <summary>
        /// 批量压缩：对已生成完毕的帧序列做一次静音段截断。
        /// 供非流式路径或事后修复使用；流式路径请用 <see cref="Advance"/> +
        /// <see cref="ShouldEmit"/> 逐帧判定。
        /// </summary>
        public static List<int[]> Collapse(List<int[]> frames)
        {
            if (frames == null || frames.Count == 0)
                return frames;

            var result = new List<int[]>(frames.Count);
            int run = 0;
            foreach (int[] frame in frames)
            {
                run = Advance(frame, run);
                if (ShouldEmit(frame, run))
                    result.Add(frame);
            }
            return result;
        }
    }
}
