from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from pydantic import Extra
import requests
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import numpy as np
from datetime import datetime
from langchain_core.output_parsers import JsonOutputParser
import json 

class TritonLLM(LLM):
    llm_url = "http://localhost:5555/v2/models/tensorrt_llm_bls/generate"

    class Config:
        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        return "Triton LLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        payload = {
            
                "text_input": prompt,
                "parameters": {
                "max_tokens": 300,
                "bad_words":[""],
                "stop_words":[""],
                "temperature": 0.0,
                #"top_k":50,
                #"top_p":0.95,
                #"random_seed": 123,
                #"repetition_penalty":1,
                }
            
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(self.llm_url, json=payload, headers=headers)
        response.raise_for_status()

        return response.json()['text_output']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llmUrl": self.llm_url}


llm = TritonLLM()


