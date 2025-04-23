import torch


class ConditioningAverageBREAK:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The text prompt with segments separated by BREAK.",
                    },
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model used for encoding the text segments."},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = (
        "A conditioning tensor representing the average of the encoded text segments.",
    )
    FUNCTION = "encode_average"

    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes text segments separated by 'BREAK' using CLIP and averages the resulting conditioning tensors."

    def encode_average(self, clip, text):
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model."
            )

        text_segments = [s.strip() for s in text.split("BREAK") if s.strip()]

        if not text_segments or len(text_segments) == 1:
            tokens = clip.tokenize(text_segments[0])
            return (clip.encode_from_tokens_scheduled(tokens),)

        conditionings = []
        pooled_outputs = []
        max_len = 0

        # Encode each segment and find max length
        for segment in text_segments:
            tokens = clip.tokenize(segment)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conditionings.append(cond)
            pooled_outputs.append(pooled)
            if cond.shape[1] > max_len:
                max_len = cond.shape[1]

        # Pad and average conditioning tensors
        avg_cond = torch.zeros_like(conditionings[0][:, :max_len, :])
        valid_pooled_count = 0
        avg_pooled = None

        for i, cond in enumerate(conditionings):
            padded_cond = cond
            if cond.shape[1] < max_len:
                padding = torch.zeros(
                    (cond.shape[0], max_len - cond.shape[1], cond.shape[2]),
                    device=cond.device,
                    dtype=cond.dtype,
                )
                padded_cond = torch.cat([cond, padding], dim=1)
            avg_cond += padded_cond

            # Average pooled outputs if they exist
            if pooled_outputs[i] is not None:
                if avg_pooled is None:
                    avg_pooled = torch.zeros_like(pooled_outputs[i])
                avg_pooled += pooled_outputs[i]
                valid_pooled_count += 1

        avg_cond /= len(conditionings)
        if avg_pooled is not None and valid_pooled_count > 0:
            avg_pooled /= valid_pooled_count

        pooled_dict = {}
        if avg_pooled is not None:
            pooled_dict["pooled_output"] = avg_pooled

        return ([(avg_cond, pooled_dict)],)


NODE_CLASS_MAPPINGS = {
    "ConditioningAverageBREAK": ConditioningAverageBREAK,
}
