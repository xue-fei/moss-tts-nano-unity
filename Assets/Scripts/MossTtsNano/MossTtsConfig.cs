using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// 语音预设数据
    /// </summary>
    [Serializable]
    public class VoicePreset
    {
        public string Name;
        public string PromptAudioPath;
        public string Description;
    }

    /// <summary>
    /// 生成参数配置
    /// </summary>
    [Serializable]
    public class GenerationConfig
    {
        public int max_new_frames = 375;
        public bool do_sample = true;
        public string sample_mode = "fixed";
        public float text_temperature = 1.0f;
        public float text_top_p = 1.0f;
        public int text_top_k = 50;
        public float audio_temperature = 0.8f;
        public float audio_top_p = 0.95f;
        public int audio_top_k = 25;
        public float audio_repetition_penalty = 1.2f;
    }

    /// <summary>
    /// TTS 配置（从 manifest 加载）
    /// </summary>
    [Serializable]
    public class TtsConfig
    {
        public int n_vq = 16;
        public int audio_pad_token_id = 1024;
        public int pad_token_id = 3;
        public int im_start_token_id = 4;
        public int im_end_token_id = 5;
        public int audio_start_token_id = 6;
        public int audio_end_token_id = 7;
        public int audio_user_slot_token_id = 8;
        public int audio_assistant_slot_token_id = 9;
        public int[] audio_codebook_sizes = Enumerable.Repeat(1024, 16).ToArray();
        public int vocab_size = 16384;
    }

    /// <summary>
    /// 合成结果
    /// </summary>
    [Serializable]
    public class SynthesisResult
    {
        public string AudioPath;
        public float[] Waveform;
        public int SampleRate;
        public int Channels;
        public float ElapsedSeconds;
        public string Voice;
        public string Mode;
        public string[] TextChunks;
    }

    /// <summary>
    /// 音频片段事件（流式）
    /// </summary>
    [Serializable]
    public class AudioChunkEvent
    {
        public float[] Waveform;
        public int SampleRate;
        public int ChunkIndex;
        public bool IsPause;
        public float EmittedAudioSeconds;
        public float LeadSeconds;
    }

    /// <summary>
    /// 内置语音列表
    /// </summary>
    [Serializable]
    public class BuiltinVoice
    {
        public string voice;
        public int[][] prompt_audio_codes;
    }

    /// <summary>
    /// 模型文件清单
    /// </summary>
    [Serializable]
    public class ModelManifest
    {
        public ModelFiles model_files;
        public TtsConfig tts_config;
        public GenerationConfig generation_defaults;
        public PromptTemplates prompt_templates;
        public BuiltinVoice[] builtin_voices;
        public TextSample[] text_samples;
    }

    [Serializable]
    public class ModelFiles
    {
        public string tts_meta;
        public string codec_meta;
        public string tokenizer_model;
    }

    [Serializable]
    public class PromptTemplates
    {
        public int[] user_prompt_prefix_token_ids;
        public int[] user_prompt_after_reference_token_ids;
        public int[] assistant_prompt_prefix_token_ids;
    }

    [Serializable]
    public class TextSample
    {
        public string text;
        public int[] text_token_ids;
    }

    /// <summary>
    /// TTS 模型元数据
    /// </summary>
    [Serializable]
    public class TtsModelMeta
    {
        public TtsFiles files;
        public TtsModelConfig model_config;
        public OnnxMetadata onnx;
    }

    [Serializable]
    public class TtsFiles
    {
        public string prefill;
        public string decode_step;
        public string local_decoder;
        public string local_greedy_frame;
        public string local_fixed_sampled_frame;
        public string local_cached_step;
    }

    [Serializable]
    public class TtsModelConfig
    {
        public int n_vq = 16;
        public int row_width = 17;
        public int hidden_size = 768;
        public int global_layers = 12;
        public int global_heads = 12;
        public int head_dim = 64;
        public int local_layers = 1;
        public int local_heads = 12;
        public int local_head_dim = 64;
        public int vocab_size = 16384;
        public int[] audio_codebook_sizes = Enumerable.Repeat(1024, 16).ToArray();
        public int audio_pad_token_id = 1024;
        public int pad_token_id = 3;
        public int im_start_token_id = 4;
        public int im_end_token_id = 5;
        public int audio_start_token_id = 6;
        public int audio_end_token_id = 7;
        public int audio_user_slot_token_id = 8;
        public int audio_assistant_slot_token_id = 9;
    }

    [Serializable]
    public class OnnxMetadata
    {
        public string[] prefill_output_names;
        public string[] decode_input_names;
        public string[] decode_output_names;
        public string[] local_cached_input_names;
        public string[] local_cached_output_names;
        public string[] local_fixed_sampled_frame_input_names;
        public string[] local_fixed_sampled_frame_output_names;
    }

    /// <summary>
    /// Codec 模型元数据
    /// </summary>
    [Serializable]
    public class CodecModelMeta
    {
        public CodecFiles files;
        public CodecConfig codec_config;
        public StreamingDecodeConfig streaming_decode;
    }

    [Serializable]
    public class CodecFiles
    {
        public string encode;
        public string decode_full;
        public string decode_step;
    }

    [Serializable]
    public class CodecConfig
    {
        public int sample_rate = 48000;
        public int channels = 2;
        public int downsample_rate = 3840;
        public int num_quantizers = 16;
    }

    [Serializable]
    public class StreamingDecodeConfig
    {
        public TransformerOffset[] transformer_offsets;
        public AttentionCache[] attention_caches;
    }

    [Serializable]
    public class TransformerOffset
    {
        public string input_name;
        public string output_name;
        public int[] shape;
    }

    [Serializable]
    public class AttentionCache
    {
        public string offset_input_name;
        public string offset_output_name;
        public int[] offset_shape;
        public string cached_keys_input_name;
        public string cached_keys_output_name;
        public string cached_values_input_name;
        public string cached_values_output_name;
        public string cached_positions_input_name;
        public string cached_positions_output_name;
        public int[] cache_shape;
        public int[] positions_shape;
    }
}
