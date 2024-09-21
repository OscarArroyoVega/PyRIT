# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from abc import abstractmethod
from typing import Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI, BadRequestError
from openai.types.chat import ChatCompletion

from pyrit.auth.azure_auth import get_token_provider_from_default_azure_credential
from pyrit.common import default_values
from pyrit.exceptions import EmptyResponseException, PyritException
from pyrit.exceptions import pyrit_target_retry, handle_bad_request_exception
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage, PromptRequestPiece, PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAIChatInterface(PromptChatTarget):

    _top_p: float
    _deployment_name: str
    _temperature: float
    _frequency_penalty: float
    _presence_penalty: float
    _client: OpenAI
    _async_client: AsyncOpenAI

    def __init__(
        self,
        *,
        memory: MemoryInterface = None,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        super().__init__(memory=memory, max_requests_per_minute=max_requests_per_minute)

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request: PromptRequestPiece = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)
        messages.append(request.to_chat_message())

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        try:
            logger.info('Starting to process the request')
            resp_text = await self._complete_chat_async(
                messages=messages,
                top_p=self._top_p,
                temperature=self._temperature,
                frequency_penalty=self._frequency_penalty,
                presence_penalty=self._presence_penalty,
            )

            logger.info(f'Received the following response from the prompt target "{resp_text}"')
            response_entry = construct_response_from_request(request=request, response_text_pieces=[resp_text])
        except BadRequestError as bre:
            response_entry = handle_bad_request_exception(response_text=bre.message, request=request)
            
        logger.info('Finished processing the request')
        return response_entry

    def _parse_chat_completion(self, response):
        """
        Parses chat message to get response

        Args:
            response (ChatMessage): The chat messages object containing the generated response message

        Returns:
            str: The generated response message
        """
        logger.info('Parsing chat completion response')
        response_message = response.choices[0].message.content
        logger.info(f'Parsed response message: {response_message}')
        return response_message

    @pyrit_target_retry
    async def _complete_chat_async(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> str:
        logger.info('Starting async chat completion')
        logger.debug(f'Parameters - max_tokens: {max_tokens}, temperature: {temperature}, top_p: {top_p}, frequency_penalty: {frequency_penalty}, presence_penalty: {presence_penalty}')   

        response: ChatCompletion = await self._async_client.chat.completions.create(
            model=self._deployment_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            stream=False,
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],  # type: ignore
        )
        finish_reason = response.choices[0].finish_reason
        extracted_response: str = ""
        # finish_reason="stop" means API returned complete message and
        # "length" means API returned incomplete message due to max_tokens limit.
        if finish_reason in ["stop", "length"]:
            extracted_response = self._parse_chat_completion(response)
            # Handle empty response
            if not extracted_response:
                raise EmptyResponseException(message="The chat returned an empty response.")
        else:
            raise PyritException(message=f"Unknown finish_reason {finish_reason}")
        logger.info('Async chat completion finished')
        return extracted_response

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")


class UnifyChatTarget(OpenAIChatInterface):
    API_KEY_ENVIRONMENT_VARIABLE: str = "UNIFY_CHAT_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "UNIFY_CHAT_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "UNIFY_CHAT_DEPLOYMENT"

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        memory: MemoryInterface = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
        headers: Optional[dict[str, str]] = None,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
       
        super().__init__(memory=memory, max_requests_per_minute=max_requests_per_minute)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._deployment_name = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        if headers:
            self._client = OpenAI(api_key=api_key, 
                                  base_url=endpoint,
                                  default_headers=json.loads(str(headers)))
        else:
            self._client = OpenAI(
                api_key=api_key,
                base_url=endpoint,
            )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
        )