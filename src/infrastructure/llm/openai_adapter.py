import logging
from typing import List, Optional
from src.domain.ports import ILLMProvider
from src.domain.models import ChatMessage, DocumentChunk, LLMResponse

logger = logging.getLogger(__name__)

class OpenAIAdapter(ILLMProvider):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        enable_reasoning: bool = False
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_reasoning = enable_reasoning
        self._client = None

        if not self.api_key:
            logger.warning("OpenAI API key is not configured. OPENAI_API_KEY is required.")
        else:
            try:
                import openai
                kwargs = {"api_key": self.api_key}
                if self.base_url and str(self.base_url).strip():
                    kwargs["base_url"] = str(self.base_url).strip()
                self._client = openai.OpenAI(**kwargs)
            except Exception as e:
                logger.error("Failed to initialize OpenAI client: %s", e)
                self._client = None

    def generate_response(
        self,
        messages: List[ChatMessage],
        context_chunks: List[DocumentChunk],
        system_prompt: str
    ) -> LLMResponse:
        if not self._client:
            raise RuntimeError(
                "OpenAI API client is not initialized. Please ensure OPENAI_API_KEY is set in your .env file."
            )

        context_str = "\n\n".join([f"Document [{c.doc_name}]:\n{c.content}" for c in context_chunks])
        full_system_prompt = f"{system_prompt}\n\nRELEVANT KNOWLEDGE BASE CONTEXT:\n{context_str}"

        formatted_messages = [{"role": "system", "content": full_system_prompt}]
        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

        create_kwargs = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            create_kwargs["max_tokens"] = self.max_tokens
        if "json" in system_prompt.lower():
            create_kwargs["response_format"] = {"type": "json_object"}
        if self.enable_reasoning:
            create_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        try:
            response = self._client.chat.completions.create(**create_kwargs)
        except Exception as e:
            # Fallback if extra_body or response_format is unsupported by target provider/model
            if "extra_body" in create_kwargs or "response_format" in create_kwargs:
                create_kwargs.pop("extra_body", None)
                create_kwargs.pop("response_format", None)
                try:
                    response = self._client.chat.completions.create(**create_kwargs)
                except Exception as inner_e:
                    logger.error("OpenAI API call failed for model '%s': %s", self.model_name, inner_e)
                    raise RuntimeError(f"OpenAI API Error: {str(inner_e)}") from inner_e
            else:
                logger.error("OpenAI API call failed for model '%s': %s", self.model_name, e)
                raise RuntimeError(f"OpenAI API Error: {str(e)}") from e

        content = response.choices[0].message.content
        usage = response.usage
        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model_name=self.model_name
        )
