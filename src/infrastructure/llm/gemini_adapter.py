import logging
from typing import List, Optional
from src.domain.ports import ILLMProvider
from src.domain.models import ChatMessage, DocumentChunk, LLMResponse

logger = logging.getLogger(__name__)

class GeminiAdapter(ILLMProvider):
    def __init__(
        self,
        model_name: str = "gemini-3.6-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        enable_reasoning: bool = False
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_reasoning = enable_reasoning
        self._client = None

        if not self.api_key:
            logger.warning("Gemini API key is not configured. GEMINI_API_KEY is required for Google provider.")
        else:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize Google GenAI client: %s", e)
                self._client = None

    def generate_response(
        self,
        messages: List[ChatMessage],
        context_chunks: List[DocumentChunk],
        system_prompt: str
    ) -> LLMResponse:
        if not self._client:
            raise RuntimeError(
                "Google Gemini client is not initialized. Please ensure GEMINI_API_KEY is configured in your environment."
            )

        context_str = "\n\n".join([f"Document [{c.doc_name}]:\n{c.content}" for c in context_chunks])
        full_system_instruction = f"{system_prompt}\n\nRELEVANT KNOWLEDGE BASE CONTEXT:\n{context_str}"

        # Build native multi-turn input list for Google Interactions API
        interaction_input = []
        for msg in messages:
            msg_type = "user_input" if msg.role.lower() in ["user", "human"] else "model_output"
            interaction_input.append({
                "type": msg_type,
                "content": [{"type": "text", "text": msg.content}]
            })

        user_content = "\n".join([f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in messages])

        try:
            # 1. Try Google Interactions API (client.interactions.create)
            if hasattr(self._client, 'interactions'):
                try:
                    gen_config = {}
                    if self.max_tokens:
                        gen_config["max_output_tokens"] = self.max_tokens

                    if not self.enable_reasoning:
                        gen_config["thinking_level"] = "minimal"
                        gen_config["thinking_summaries"] = "none"
                    else:
                        gen_config["thinking_level"] = "medium"
                        gen_config["thinking_summaries"] = "auto"

                    create_kwargs = {
                        "model": self.model_name,
                        "input": interaction_input if interaction_input else user_content,
                        "system_instruction": full_system_instruction,
                    }
                    if gen_config:
                        create_kwargs["generation_config"] = gen_config

                    if "json" in system_prompt.lower():
                        create_kwargs["response_format"] = {
                            "type": "text",
                            "mime_type": "application/json"
                        }

                    interaction = self._client.interactions.create(**create_kwargs)
                    output_text = getattr(interaction, 'output_text', None)
                    if output_text:
                        prompt_tokens = 0
                        completion_tokens = 0
                        total_tokens = 0

                        if hasattr(interaction, 'usage') and interaction.usage:
                            u = interaction.usage
                            if isinstance(u, dict):
                                prompt_tokens = u.get("total_input_tokens", 0) or u.get("input_tokens", 0) or 0
                                completion_tokens = u.get("total_output_tokens", 0) or u.get("output_tokens", 0) or 0
                                total_tokens = u.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
                            else:
                                prompt_tokens = getattr(u, 'total_input_tokens', 0) or getattr(u, 'input_tokens', 0) or getattr(u, 'prompt_tokens', 0) or 0
                                completion_tokens = getattr(u, 'total_output_tokens', 0) or getattr(u, 'output_tokens', 0) or getattr(u, 'completion_tokens', 0) or 0
                                total_tokens = getattr(u, 'total_tokens', 0) or (prompt_tokens + completion_tokens)
                        elif hasattr(interaction, 'usage_metadata') and interaction.usage_metadata:
                            prompt_tokens = getattr(interaction.usage_metadata, 'prompt_token_count', 0) or 0
                            completion_tokens = getattr(interaction.usage_metadata, 'candidates_token_count', 0) or 0
                            total_tokens = getattr(interaction.usage_metadata, 'total_token_count', 0) or (prompt_tokens + completion_tokens)
                        else:
                            prompt_tokens = getattr(interaction, 'prompt_tokens', 0) or getattr(interaction, 'input_tokens', 0) or 0
                            completion_tokens = getattr(interaction, 'completion_tokens', 0) or getattr(interaction, 'output_tokens', 0) or 0
                            total_tokens = getattr(interaction, 'total_tokens', 0) or (prompt_tokens + completion_tokens)

                        if total_tokens == 0:
                            prompt_tokens = len(full_system_instruction.split()) + sum(len(m.content.split()) for m in messages)
                            completion_tokens = len(output_text.split())
                            total_tokens = prompt_tokens + completion_tokens

                        return LLMResponse(
                            content=output_text,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            model_name=self.model_name
                        )
                except Exception as int_err:
                    logger.debug("Interactions API call failed, falling back to generate_content: %s", int_err)

            # 2. Fallback to client.models.generate_content
            from google.genai import types

            config_kwargs = {
                "system_instruction": full_system_instruction,
                "temperature": self.temperature,
            }
            if self.max_tokens:
                config_kwargs["max_output_tokens"] = self.max_tokens

            if "json" in system_prompt.lower():
                config_kwargs["response_mime_type"] = "application/json"

            config = types.GenerateContentConfig(**config_kwargs)

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=config
            )

            text_content = response.text or ""

            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
                total_tokens = getattr(response.usage_metadata, 'total_token_count', 0) or (prompt_tokens + completion_tokens)

            return LLMResponse(
                content=text_content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model_name=self.model_name
            )
        except Exception as e:
            logger.error("Gemini API call failed for model '%s': %s", self.model_name, e)
            raise RuntimeError(f"Gemini API Error: {str(e)}") from e
