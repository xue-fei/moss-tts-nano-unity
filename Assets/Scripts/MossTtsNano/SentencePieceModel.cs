using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MossTtsNano
{
    /// <summary>
    /// 直接解析 SentencePiece 的 tokenizer.model 二进制 protobuf，
    /// 彻底去掉「先跑 Python 导出 JSON」这一步。
    ///
    /// ModelProto 顶层三个字段：
    ///   field 1 (repeated SentencePiece) —— piece / score / type
    ///   field 2 (TrainerSpec)            —— model_type、byte_fallback、特殊 token id
    ///   field 3 (NormalizerSpec)         —— precompiled_charsmap 归一化规则表
    /// </summary>
    internal static class SentencePieceModel
    {
        // ==== protobuf wire types ====
        private const int WireVarint = 0;
        private const int Wire64Bit = 1;
        private const int WireLengthDelimited = 2;
        private const int Wire32Bit = 5;

        /// <summary>解析结果。</summary>
        public sealed class Result
        {
            public string[] Pieces;
            public float[] Scores;
            public int[] Types;
            public int UnkId;
            public int BosId = 1;
            public int EosId = 2;
            public int PadId = -1;
            public bool ByteFallback = true;
            public byte[] PrecompiledCharsMap;
            public bool AddDummyPrefix = true;
            public bool RemoveExtraWhitespaces = true;
            public bool EscapeWhitespaces = true;
        }

        public static Result Parse(byte[] data)
        {
            var result = new Result();
            int pos = 0;

            // 顶层 ModelProto 的三个字段
            List<byte[]> pieceMessages = new List<byte[]>();
            byte[] trainerSpec = null;
            byte[] normalizerSpec = null;

            while (pos < data.Length)
            {
                int tag = (int)ReadVarint(data, ref pos);
int field = tag >> 3;
                    int wire = tag & 0x7;

                    switch (wire)
                    {
                        case WireVarint:
                            ReadVarint(data, ref pos);
                            break;
                        case WireLengthDelimited:
                            int len = (int)ReadVarint(data, ref pos);
                            if (field == 1)
                                pieceMessages.Add(ReadBytes(data, ref pos, len));
                            else if (field == 2)
                                trainerSpec = ReadBytes(data, ref pos, len);
                            else if (field == 3)
                                normalizerSpec = ReadBytes(data, ref pos, len);
                            else
                                pos += len;
                        break;
                    case Wire32Bit:
                        pos += 4;
                        break;
                    case Wire64Bit:
                        pos += 8;
                        break;
                    default:
                        throw new InvalidDataException($"unknown wire type {wire} at pos {pos}");
                }
            }

            ParsePieces(pieceMessages, result);
            if (trainerSpec != null)
                ParseTrainerSpec(trainerSpec, result);
            if (normalizerSpec != null)
                ParseNormalizerSpec(normalizerSpec, result);

            return result;
        }

        private static void ParsePieces(List<byte[]> messages, Result result)
        {
            int n = messages.Count;
            result.Pieces = new string[n];
            result.Scores = new float[n];
            result.Types = new int[n];

            for (int i = 0; i < n; i++)
            {
                byte[] msg = messages[i];
                int pos = 0;
                while (pos < msg.Length)
                {
                    int tag = (int)ReadVarint(msg, ref pos);
                    int field = tag >> 3;
                    int wire = tag & 0x7;

                    if (wire == WireLengthDelimited)
                    {
                        int len = (int)ReadVarint(msg, ref pos);
                        if (field == 1)
                            result.Pieces[i] = Encoding.UTF8.GetString(msg, pos, len);
                        pos += len;
                    }
                    else if (wire == Wire32Bit)
                    {
                        if (field == 2)
                            result.Scores[i] = BitConverter.ToSingle(msg, pos);
                        pos += 4;
                    }
                    else if (wire == WireVarint)
                    {
                        if (field == 3)
                            result.Types[i] = (int)ReadVarint(msg, ref pos);
                        else
                            ReadVarint(msg, ref pos);
                    }
                    else if (wire == Wire64Bit)
                    {
                        pos += 8;
                    }
                    else
                    {
                        throw new InvalidDataException($"unknown wire type {wire} in piece msg at pos {pos}");
                    }
                }
            }
        }

        private static void ParseTrainerSpec(byte[] data, Result result)
        {
            int pos = 0;
            while (pos < data.Length)
            {
                int tag = (int)ReadVarint(data, ref pos);
                int field = tag >> 3;
                int wire = tag & 0x7;

                if (wire == WireVarint)
                {
                    long val = ReadVarint(data, ref pos);
                    switch (field)
                    {
                        case 19: result.ByteFallback = val != 0; break;  // byte_fallback
                        // field 40 是 hard_vocab_limit，不是 byte_fallback！
                        // 原实现错误地用 field 40 覆盖 ByteFallback，导致 byte_fallback
                        // 被错误地设为 false，OOV 字符（如 U+200C）无法拆成 byte token。
                        case 41: result.UnkId = (int)val; break;          // unk_id
                        case 42: result.BosId = (int)val; break;          // bos_id
                        case 43: result.EosId = (int)val; break;          // eos_id
                        case 44: result.PadId = (int)val; break;          // pad_id
                        default: break;
                    }
                }
                else if (wire == WireLengthDelimited)
                {
                    int len = (int)ReadVarint(data, ref pos);
                    pos += len;
                }
                else if (wire == Wire32Bit)
                {
                    pos += 4;
                }
                else if (wire == Wire64Bit)
                {
                    pos += 8;
                }
                else
                {
                    throw new InvalidDataException($"unknown wire type {wire} in trainer_spec at pos {pos}");
                }
            }
        }

        private static void ParseNormalizerSpec(byte[] data, Result result)
        {
            int pos = 0;
            while (pos < data.Length)
            {
                int tag = (int)ReadVarint(data, ref pos);
                int field = tag >> 3;
                int wire = tag & 0x7;

                if (wire == WireLengthDelimited)
                {
                    int len = (int)ReadVarint(data, ref pos);
                    if (field == 2)
                    {
                        // precompiled_charsmap
                        result.PrecompiledCharsMap = new byte[len];
                        Buffer.BlockCopy(data, pos, result.PrecompiledCharsMap, 0, len);
                    }
                    pos += len;
                }
                else if (wire == WireVarint)
                {
                    long val = ReadVarint(data, ref pos);
                    switch (field)
                    {
                        case 3: result.AddDummyPrefix = val != 0; break;
                        case 4: result.RemoveExtraWhitespaces = val != 0; break;
                        case 5: result.EscapeWhitespaces = val != 0; break;
                        default: break;
                    }
                }
                else if (wire == Wire32Bit)
                {
                    pos += 4;
                }
                else if (wire == Wire64Bit)
                {
                    pos += 8;
                }
                else
                {
                    throw new InvalidDataException($"unknown wire type {wire} in normalizer_spec at pos {pos}");
                }
            }
        }

        // ==== protobuf 基础读取 ====

        private static long ReadVarint(byte[] data, ref int pos)
        {
            long result = 0;
            int shift = 0;
            while (pos < data.Length)
            {
                byte b = data[pos++];
                result |= (long)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) break;
                shift += 7;
            }
            return result;
        }

        private static byte[] ReadBytes(byte[] data, ref int pos, int length)
        {
            byte[] result = new byte[length];
            Buffer.BlockCopy(data, pos, result, 0, length);
            pos += length;
            return result;
        }
    }
}
