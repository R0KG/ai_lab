"""Tests for the AWS Bedrock text-generation provider."""

import boto3
import pytest

from app.llm.bedrock import BedrockLLMProvider


class FakeBedrockClient:
    """Capture a Converse request and return a deterministic response."""

    def converse(self, **kwargs):
        assert kwargs["modelId"] == "amazon.nova-lite-v1:0"
        assert kwargs["messages"] == [
            {
                "role": "user",
                "content": [{"text": "What is Python?"}],
            }
        ]
        return {
            "output": {
                "message": {
                    "content": [{"text": "  grounded answer  "}],
                }
            },
            "usage": {
                "inputTokens": 12,
                "outputTokens": 3,
            },
        }


class FakeSession:
    """Provide the fake Bedrock runtime client."""

    def client(self, service_name: str, region_name: str):
        assert service_name == "bedrock-runtime"
        assert region_name == "eu-central-1"
        return FakeBedrockClient()


@pytest.mark.asyncio
async def test_bedrock_provider_returns_normalized_response(monkeypatch) -> None:
    monkeypatch.setattr(boto3, "Session", FakeSession)
    provider = BedrockLLMProvider(
        region="eu-central-1",
        model_id="amazon.nova-lite-v1:0",
    )

    response = await provider.generate("  What is Python?  ")

    assert response.text == "grounded answer"
    assert response.model == "amazon.nova-lite-v1:0"
    assert response.input_tokens == 12
    assert response.output_tokens == 3
