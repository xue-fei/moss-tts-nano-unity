using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace MossTtsNano
{
    /// <summary>
    /// TTS 输入鲁棒性正则化器，逐规则对齐 Python tts_robust_normalizer_single_script.py。
    ///
    /// 原 C# NormalizeTtsText 只做 Trim + 去除控制字符，缺少汉字间空格删除、
    /// 中英混排边界空格等规则，导致空格被 SentencePiece 编成 ▁ token，
    /// 模型在词与词之间产生不合理的停顿空白。
    ///
    /// 规则链（与 Python 一一对应）：
    ///   BaseCleanup → NormalizeMarkdownAndLines → NormalizeFlowArrows
    ///   → ProtectSpans → NormalizeVisibleUnderscores → NormalizeSpaces
    ///   → NormalizeStructuralPunctuation → NormalizeRepeatedPunctuation
    ///   → NormalizeSpaces → RestoreSpans → EnsureTerminalPunctuationByLine
    /// </summary>
    internal static class TtsRobustNormalizer
    {
        // CJK 字符范围：汉字 ExtA + 基本汉字 + 日文假名
        private const string CjkClass = @"[\u3400-\u4DBF\u4E00-\u9FFF\u3040-\u30FF]";

        // 零宽字符
        private static readonly Regex ZeroWidthRe = new Regex(
            @"[\u200B-\u200D\uFEFF]", RegexOptions.Compiled);

        // 保护占位符
        private const string ProtTag = "___PROT";
        private static readonly Regex ProtRe = new Regex(
            @"^___PROT\d+___$", RegexOptions.Compiled);

        // 需要保护的高风险 token（按优先级排序）
        private static readonly Regex[] ProtectPatterns = {
            new Regex(@"https?://[^\s\u3000，。！？；、）】》〉」』]+", RegexOptions.Compiled),
            new Regex(@"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])", RegexOptions.Compiled),
            new Regex(@"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}", RegexOptions.Compiled),
            new Regex(@"(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+", RegexOptions.Compiled),
            new Regex(@"(?<![A-Za-z0-9_])#(?!\s)[^\s#]+", RegexOptions.Compiled),
            new Regex(@"(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+", RegexOptions.Compiled),
            new Regex(@"(?<![A-Za-z0-9_])(?=[A-Za-z0-9._/+:-]*[A-Za-z])(?=[A-Za-z0-9._/+:-]*[./+:-])[A-Za-z0-9][A-Za-z0-9._/+:-]*(?![A-Za-z0-9_])", RegexOptions.Compiled),
        };

        // 拉丁类 token：与 Python _LATINISH 对齐
        // 保护占位符 或 以字母数字开头且至少含一个拉丁字母的连续 token
        private const string LatinishPattern =
            @"(?:___PROT\d+___|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)";

        // 中文标点
        private static readonly Regex ChinesePunctCloseRe = new Regex(
            @"\s+([，。！？；：、"")」』】）》])", RegexOptions.Compiled);
        private static readonly Regex ChinesePunctOpenRe = new Regex(
            @"([（【「『《“‘])\s+", RegexOptions.Compiled);
        private static readonly Regex ChinesePunctAfterRe = new Regex(
            @"([，。！？；：、])\s*", RegexOptions.Compiled);
        private static readonly Regex AsciiPunctBeforeRe = new Regex(
            @"\s+([,.;!?])", RegexOptions.Compiled);

        // Markdown 链接
        private static readonly Regex MdLinkRe = new Regex(
            @"\[([^\[\]]+?)\]\((https?://[^)\s]+)\)", RegexOptions.Compiled);

        // 流程箭头
        private static readonly Regex FlowArrowRe = new Regex(
            @"\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*", RegexOptions.Compiled);

        // 结构性括号
        private static readonly Regex BracketSquareRe = new Regex(
            @"\[\s*([^\[\]]+?)\s*\]", RegexOptions.Compiled);
        private static readonly Regex BracketCurlyRe = new Regex(
            @"\{\s*([^{}]+?)\s*\}", RegexOptions.Compiled);
        private static readonly Regex BracketChineseRe = new Regex(
            @"[【〖『「]\s*([^】〗』」]+?)\s*[】〗』」]", RegexOptions.Compiled);

        // 重复标点
        private static readonly Regex EllipsisRe = new Regex(
            @"(?:\.{3,}|…{2,}|……+)", RegexOptions.Compiled);
        private static readonly Regex RepeatCnPeriodRe = new Regex(
            @"[。．]{2,}", RegexOptions.Compiled);
        private static readonly Regex RepeatCnCommaRe = new Regex(
            @"[，,]{2,}", RegexOptions.Compiled);
        private static readonly Regex RepeatExclaimRe = new Regex(
            @"[!！]{2,}", RegexOptions.Compiled);
        private static readonly Regex RepeatQuestionRe = new Regex(
            @"[?？]{2,}", RegexOptions.Compiled);
        private static readonly Regex MixedQeRe = new Regex(
            @"[!?！？]{2,}", RegexOptions.Compiled);

        // 长破折号
        private static readonly Regex LongDashRe = new Regex(
            @"\s*(?:—|–|―|-){2,}\s*", RegexOptions.Compiled);

        // 尾部闭合标点
        private static readonly HashSet<char> TrailingClosers = new HashSet<char> {
            '"', '\'', ')', ']', '}', '）', '】', '》', '〉', '」', '』', '”', '’'
        };

        /// <summary>主入口，对齐 Python normalize_tts_text。</summary>
        public static string Normalize(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            text = BaseCleanup(text);
            text = NormalizeMarkdownAndLines(text);
            text = NormalizeFlowArrows(text);

            // Protect → normalize → restore
            var protectedList = new List<string>();
            text = ProtectSpans(text, protectedList);

            text = NormalizeVisibleUnderscores(text);
            text = NormalizeSpaces(text);
            text = NormalizeStructuralPunctuation(text);
            text = NormalizeRepeatedPunctuation(text);
            text = NormalizeSpaces(text);

            text = RestoreSpans(text, protectedList);
            text = text.Trim();
            text = EnsureTerminalPunctuationByLine(text);

            return text;
        }

        /// <summary>基础清理：换行统一、零宽字符删除、控制字符删除。</summary>
        private static string BaseCleanup(string text)
        {
            text = text.Replace("\r\n", "\n").Replace("\r", "\n").Replace('\u3000', ' ');
            text = ZeroWidthRe.Replace(text, "");

            var sb = new StringBuilder(text.Length);
            foreach (char c in text)
            {
                if (c == '\n' || c == '\t' || !char.IsControl(c))
                    sb.Append(c);
            }
            return sb.ToString();
        }

        /// <summary>Markdown 链接、标题、列表、换行处理。</summary>
        private static string NormalizeMarkdownAndLines(string text)
        {
            text = MdLinkRe.Replace(text, "$1 $2");

            var lines = text.Split('\n');
            var result = new List<string>();

            foreach (string raw in lines)
            {
                string line = raw.Trim();
                if (string.IsNullOrEmpty(line)) continue;

                line = Regex.Replace(line, @"^#{1,6}\s+", "");
                line = Regex.Replace(line, @"^>\s+", "");
                line = Regex.Replace(line, @"^[-*+]\s+", "");
                line = Regex.Replace(line, @"^\d+[.)]\s+", "");

                result.Add(line);
            }

            if (result.Count == 0) return "";

            // Python: 逐行确保前一行有末尾标点，然后直接拼接（无分隔符）
            var merged = new List<string> { result[0] };
            for (int i = 1; i < result.Count; i++)
            {
                int lastIdx = merged.Count - 1;
                merged[lastIdx] = EnsureTerminalPunctuation(merged[lastIdx]);
                merged.Add(result[i]);
            }

            return string.Join("", merged);
        }

        /// <summary>流程/映射箭头转中文逗号。</summary>
        private static string NormalizeFlowArrows(string text) =>
            FlowArrowRe.Replace(text, "，");

        /// <summary>保护高风险 token 不被后续规则修改。</summary>
        private static string ProtectSpans(string text, List<string> protectedList)
        {
            foreach (Regex pattern in ProtectPatterns)
            {
                text = pattern.Replace(text, match =>
                {
                    int idx = protectedList.Count;
                    protectedList.Add(match.Value);
                    return $"{ProtTag}{idx}___";
                });
            }
            return text;
        }

        /// <summary>恢复被保护的 token。</summary>
        private static string RestoreSpans(string text, List<string> protectedList)
        {
            for (int i = 0; i < protectedList.Count; i++)
            {
                text = text.Replace($"{ProtTag}{i}___", protectedList[i]);
            }
            return text;
        }

        /// <summary>非保护区域中的可见下划线替换为空格。</summary>
        private static string NormalizeVisibleUnderscores(string text)
        {
            var parts = Regex.Split(text, @"(___PROT\d+___)");
            for (int i = 0; i < parts.Length; i++)
            {
                if (!IsProtected(parts[i]))
                    parts[i] = parts[i].Replace('_', ' ');
            }
            return string.Join("", parts);
        }

        /// <summary>
        /// 空格规则：英文压缩、中文删除、中英混排保留边界。
        /// 这是修复「词间不合理空白」的核心规则。
        /// </summary>
        private static string NormalizeSpaces(string text)
        {
            // 统一空白
            text = Regex.Replace(text, @"[ \t\r\f\v]+", " ");

            // 汉字之间：删除空格（核心规则，直接解决词间空白问题）
            text = Regex.Replace(text, $@"(?<ch>{CjkClass})\s+(?={CjkClass})", "${ch}");

            // 汉字与纯数字之间：删除空格
            text = Regex.Replace(text, $@"(?<ch>{CjkClass})\s+(?=\d)", "${ch}");
            text = Regex.Replace(text, @"(?<d>\d)\s+(?={CjkClass})", "${d}");

            // 汉字与拉丁类 token（含保护占位符）相邻：保留/补 1 个空格
            text = Regex.Replace(text, $@"(?<ch>{CjkClass})(?={LatinishPattern})", "${ch} ");
            text = Regex.Replace(text, $@"(?<latin>{LatinishPattern})(?={CjkClass})", "${latin} ");

            // 再压连续空格
            text = Regex.Replace(text, @" {2,}", " ");

            // 中文标点前后不保留空格
            text = ChinesePunctCloseRe.Replace(text, "$1");
            text = ChinesePunctOpenRe.Replace(text, "$1");
            text = ChinesePunctAfterRe.Replace(text, "$1");

            // ASCII 标点前不留空格
            text = AsciiPunctBeforeRe.Replace(text, "$1");

            text = Regex.Replace(text, @" {2,}", " ");
            return text.Trim();
        }

        /// <summary>结构性括号统一转双引号包裹内容，长破折号转句边界。</summary>
        private static string NormalizeStructuralPunctuation(string text)
        {
            text = BracketSquareRe.Replace(text, "\"$1\"");
            text = BracketCurlyRe.Replace(text, "\"$1\"");
            text = BracketChineseRe.Replace(text, "\"$1\"");

            // 《》独立标题处理
            text = Regex.Replace(text,
                @"(^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))",
                "$1$2");

            text = NormalizeFlowArrows(text);
            text = LongDashRe.Replace(text, "。");

            return text;
        }

        /// <summary>重复标点收敛。</summary>
        private static string NormalizeRepeatedPunctuation(string text)
        {
            text = EllipsisRe.Replace(text, "。");
            text = RepeatCnPeriodRe.Replace(text, "。");
            text = RepeatCnCommaRe.Replace(text, "，");
            text = RepeatExclaimRe.Replace(text, "！");
            text = RepeatQuestionRe.Replace(text, "？");

            text = MixedQeRe.Replace(text, match =>
            {
                string s = match.Value;
                bool hasQ = s.Contains('?') || s.Contains('？');
                bool hasE = s.Contains('!') || s.Contains('！');
                if (hasQ && hasE) return "？！";
                return hasQ ? "？" : "！";
            });

            return text;
        }

        /// <summary>确保单行末尾有标点。</summary>
        private static string EnsureTerminalPunctuation(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            int index = text.Length - 1;
            while (index >= 0 && char.IsWhiteSpace(text[index])) index--;
            while (index >= 0 && TrailingClosers.Contains(text[index])) index--;

            if (index >= 0 && char.IsPunctuation(text[index])) return text;
            return text + "。";
        }

        /// <summary>逐行确保末尾有标点。</summary>
        private static string EnsureTerminalPunctuationByLine(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            var lines = text.Split('\n');
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                if (!string.IsNullOrEmpty(line))
                    lines[i] = EnsureTerminalPunctuation(line);
            }
            return string.Join("\n", lines).Trim();
        }

        /// <summary>判断字符串是否为保护占位符。</summary>
        private static bool IsProtected(string s) =>
            !string.IsNullOrEmpty(s) && ProtRe.IsMatch(s);
    }
}
