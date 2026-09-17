import asyncio,copy,unittest
from unittest.mock import AsyncMock
from runtime_chat_effort import ChatEffortTest
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest,ResponsesRequest
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.entrypoints.openai.responses_compat import ToolRegistry
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
SCHEMA={'type':'object','properties':{'q':{'type':'string'}},'required':['q'],'additionalProperties':False}
def function(flag='missing',name='search'):
 f={'name':name,'parameters':copy.deepcopy(SCHEMA)}
 if flag!='missing':f['strict']=flag
 return f
class StrictTest(unittest.TestCase):
 setUpClass=classmethod(ChatEffortTest.setUpClass.__func__)
 def setUp(self):
  ChatEffortTest.setUp(self)
  self.chat.tokenizer_manager.model_config.hf_config.model_type='qwen3_8_flash_next'
  self.chat.tool_call_parser='qwen3_coder'
  self.responses=OpenAIServingResponses.__new__(OpenAIServingResponses)
  self.responses.__dict__.update(self.chat.__dict__)
 def test_chat_conversion_constraints_and_immutable_opt_out(self):
  for model in ('qwen3_8_flash_next','qwen3_8_flash_next_text','llama'):
   self.chat.tokenizer_manager.model_config.hf_config.model_type=model
   for flag in ('missing',False,True):
    for choice in ('auto','none'):
     with self.subTest(model=model,flag=flag,choice=choice):
      req=ChatCompletionRequest(model='alias',messages=[{'role':'user','content':'Hi'}],tools=[{'type':'function','function':function(flag)}],tool_choice=choice)
      before=req.model_dump();fields=set(req.tools[0].function.model_fields_set)
      result=self.chat._process_messages(req,False)
      expected=choice=='auto' and (flag is True or (flag=='missing' and model!='llama'))
      self.assertEqual(result.tool_call_constraint is not None,expected)
      if model != 'llama': self.assertEqual(req.model_dump(),before)
      self.assertEqual(req.model_dump()['tools'],before['tools'])
      self.assertEqual(req.tools[0].function.model_fields_set,fields)
 def test_message_local_tools(self):
  for role in ('system','developer'):
   req=ChatCompletionRequest(model='alias',messages=[{'role':role,'content':'Tools','tools':[{'type':'function','function':function()}]},{'role':'user','content':'Hi'}],tool_choice='auto')
   # Pretokenized path tests grammar independently of template role support.
   req.input_ids=[1,2,3]
   before=req.model_dump();out=self.chat._default_qwen_tool_strictness(req)
   self.assertTrue(out.messages[0].tools[0].function.strict)
   self.assertIsNotNone(self.chat._process_messages(req,False).tool_call_constraint)
   self.assertEqual(req.model_dump(),before)
 def test_responses_before_registry(self):
  for model in ('qwen3_8_flash_next','llama'):
   self.responses.tokenizer_manager.model_config.hf_config.model_type=model
   for flag in ('missing',False,True):
    for namespace in (False,True):
     f={'type':'function',**function(flag)}
     tools=[{'type':'namespace','name':'web','tools':[f]}] if namespace else [f]
     req=ResponsesRequest(model='alias',input='Hi',tools=tools)
     before=req.model_dump();out=self.responses._default_qwen_response_tool_strictness(req)
     registry=ToolRegistry(out.tools)
     self.assertEqual(registry.functions[0]['strict'],flag is True or(flag=='missing' and model!='llama'))
     self.assertEqual(req.model_dump(),before)
 def test_responses_entrypoint_normalizes_before_dump(self):
  from fastapi.responses import ORJSONResponse
  self.responses._create_responses_internal=AsyncMock(return_value=ORJSONResponse({}))
  req=ResponsesRequest(model='alias',input='Hi',tools=[{'type':'function',**function()}])
  asyncio.run(self.responses.create_responses(req))
  internal=self.responses._create_responses_internal.call_args.args[0]
  self.assertTrue(internal.tools[0].strict)
  self.assertNotIn('strict',req.tools[0].model_fields_set)
 def test_messages_uses_shared_normalization(self):
  req=AnthropicMessagesRequest(model='alias',max_tokens=64,messages=[{'role':'user','content':'Hi'}],tools=[{'name':'search','input_schema':SCHEMA}])
  chat=AnthropicServing(self.chat)._convert_to_chat_completion_request(req)
  self.assertNotIn('strict',chat.tools[0].function.model_fields_set)
  self.assertIsNotNone(self.chat._process_messages(chat,False).tool_call_constraint)
 def test_mixed_strictness_and_idempotence(self):
  req=ChatCompletionRequest(model='alias',messages=[{'role':'user','content':'Hi'}],tools=[{'type':'function','function':function(flag,str(i))} for i,flag in enumerate(('missing',False,True))])
  out=self.chat._default_qwen_tool_strictness(req)
  self.assertEqual([t.function.strict for t in out.tools],[True,False,True])
  self.assertEqual(out.model_dump(),self.chat._default_qwen_tool_strictness(out).model_dump())
if __name__=='__main__':unittest.main(verbosity=2)
