"""AWS Bedrock provider adapter."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.llm.base import LLMProviderError, LLMResponse


@dataclass
class BedrockLLMProvider:
    """Generate text using the AWS Bedrock Converse API."""

    region: str
    model_id: str
    profile: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0

    def _converse(self, prompt: str) -> dict[str, Any]:
        """Call the synchronous boto3 Bedrock client"""
        if self.profile:
            session = boto3.Session(profile_name=self.profile)
        else:
            session = boto3.Session()

        client = session.client(
            "bedrock-runtime",
            region_name=self.region,
        )
        return client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt,
                        }
                    ],
                }
            ],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate a non-streaming response from Bedrock."""

        normalized_prompt = prompt.strip()

        if not normalized_prompt:
            raise ValueError("prompt must not be empty")

        started_at = time.perf_counter()

        try:
            response = await asyncio.to_thread(
                self._converse,
                normalized_prompt,
            )
        except (BotoCoreError, ClientError) as exc:
            raise LLMProviderError(
                "AWS Bedrock failed to generate an answer"
            ) from exc

        output = response.get("output")

        if not isinstance(output, dict):
            raise LLMProviderError(
                "AWS Bedrock response does not contain output"
            )

        message = output.get("message")

        if not isinstance(message, dict):
            raise LLMProviderError(
                "AWS Bedrock response does not contain a message"
            )

        content = message.get("content")

        if not isinstance(content, list):
            raise LLMProviderError(
                "AWS Bedrock response does not contain content"
            )

        text_blocks: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                text_blocks.append(text)

        if not text_blocks:
            raise LLMProviderError(
                "AWS Bedrock response does not contain text"
            )
        usage = response.get("usage")

        usage_data = usage if isinstance(usage, dict) else {}

        input_tokens = usage_data.get("inputTokens")
        output_tokens = usage_data.get("outputTokens")

        return LLMResponse(
            text="".join(text_blocks).strip(),
            model=self.model_id,
            input_tokens=(input_tokens if isinstance(input_tokens, int) else None),
            output_tokens=(output_tokens if isinstance(output_tokens, int) else None),
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )
