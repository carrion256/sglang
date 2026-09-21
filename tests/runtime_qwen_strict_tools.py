import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, Tool, ToolChoice
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.function_call.function_call_parser import FunctionCallParser

@pytest.mark.parametrize('strict',[None,True,False])
@pytest.mark.parametrize('choice',['auto','required','named','none'])
def test_tools_no_constraint_and_no_strict_rewrite(strict,choice):
    function={'name':'probe','parameters':{'type':'object','properties':{'query':{'type':'string','pattern':'^[^-]'}},'required':['query']}}
    if strict is not None:function['strict']=strict
    t=Tool.model_validate({'type':'function','function':function})
    request=ChatCompletionRequest(model='test',messages=[{'role':'user','content':'test'}],tools=[t],input_ids=[1,2],tool_choice=ToolChoice(function={'name':'probe'}) if choice=='named' else choice)
    serving=OpenAIServingChat.__new__(OpenAIServingChat)
    serving.tokenizer_manager=NS(server_args=NS(grammar_backend='none'),tokenizer=None,model_config=NS(hf_config=NS(model_type='qwen3_8_flash_next')))
    serving.default_chat_template_kwargs={};serving.is_gpt_oss=False;serving.is_gemma4=False
    serving.reasoning_parser=None;serving.tool_call_parser='qwen3_coder';serving.chat_encoding_spec=None
    serving._patch_reasoning_skip_special_tokens=Mock();serving._get_reasoning_from_request=Mock(return_value=False)
    before=t.model_dump();fields=t.function.model_fields_set.copy()
    result=serving._process_messages(request,False)
    assert result.tool_call_constraint is None
    assert t.model_dump()==before and t.function.model_fields_set==fields
    assert not hasattr(serving,'_default_qwen_tool_strictness')
    assert result.prompt_ids==[1,2]

@pytest.mark.parametrize('field',['json_schema','regex','ebnf','structural_tag',None])
def test_disabled_accepts_without_queue_compile_or_abort(field):
    manager=GrammarManager.__new__(GrammarManager)
    manager.server_args=NS(grammar_backend='none');manager.grammar_queue=[]
    manager.grammar_backend=Mock();manager._enable_strict_thinking=False
    params=dict(json_schema=None,regex=None,ebnf=None,structural_tag=None)
    if field:params[field]='synthetic constraint'
    req=NS(sampling_params=NS(**params),grammar=None,set_finish_with_abort=Mock())
    assert manager.process_req_with_grammar(req) is False
    assert manager.grammar_queue==[] and req.grammar is None
    req.set_finish_with_abort.assert_not_called();manager.grammar_backend.get_cached_or_future_value.assert_not_called()


def test_enabled_still_compiles():
    manager=GrammarManager.__new__(GrammarManager)
    manager.server_args=NS(grammar_backend='xgrammar');manager.grammar_queue=[]
    manager.grammar_backend=Mock();manager.grammar_backend.get_cached_or_future_value.return_value=(None,False)
    manager._enable_strict_thinking=False
    req=NS(sampling_params=NS(json_schema='{}',regex=None,ebnf=None,structural_tag=None),grammar=None,require_reasoning=False,set_finish_with_abort=Mock())
    assert manager.process_req_with_grammar(req) is True
    assert manager.grammar_queue==[req]
    manager.grammar_backend.get_cached_or_future_value.assert_called_once_with(('json','{}'),False)


def test_implicit_backend_failure_still_aborts():
    manager=GrammarManager.__new__(GrammarManager);manager.server_args=NS(grammar_backend='xgrammar')
    manager.grammar_backend=None;manager.grammar_queue=[];manager._enable_strict_thinking=False
    req=NS(sampling_params=NS(json_schema='{}',regex=None,ebnf=None,structural_tag=None),set_finish_with_abort=Mock())
    assert manager.process_req_with_grammar(req) is False
    req.set_finish_with_abort.assert_called_once()


def test_parser_still_runs():
    t=Tool.model_validate({'type':'function','function':{'name':'probe','parameters':{'type':'object','properties':{'query':{'type':'string'}}}}})
    raw='<tool_call>\n<function=probe>\n<parameter=query>nginx</parameter>\n</function>\n</tool_call>'
    parser=FunctionCallParser([t],'qwen3_coder');normal,calls=parser.parse_non_stream(raw)
    assert normal=='' and calls[0].name=='probe' and json.loads(calls[0].parameters)=={'query':'nginx'}
    parser=FunctionCallParser([t],'qwen3_coder');name=None;arguments=''
    for char in raw:
        text,events=parser.parse_stream_chunk(char)
        assert not text.strip()
        for event in events:
            name=event.name or name;arguments+=event.parameters or ''
    assert name=='probe' and json.loads(arguments)=={'query':'nginx'}

@pytest.mark.parametrize('strict',[None,False,True])
@pytest.mark.parametrize('namespace',[False,True])
def test_responses_preserves_client_strictness(strict,namespace):
    import asyncio
    from unittest.mock import AsyncMock
    from fastapi.responses import ORJSONResponse
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
    from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
    from sglang.srt.entrypoints.openai.responses_compat import ToolRegistry
    tool={'type':'function','name':'probe','parameters':{'type':'object','properties':{}}}
    if strict is not None:tool['strict']=strict
    tools=[{'type':'namespace','name':'testing','tools':[tool]}] if namespace else [tool]
    req=ResponsesRequest(model='test',input='Hi',tools=tools)
    before=req.model_dump()
    serving=OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving._create_responses_internal=AsyncMock(return_value=ORJSONResponse({}))
    asyncio.run(serving.create_responses(req))
    internal=serving._create_responses_internal.call_args.args[0]
    assert ToolRegistry(internal.tools).functions[0]['strict'] is (strict is True)
    assert req.model_dump()==before
    assert not hasattr(serving,'_default_qwen_response_tool_strictness')


def test_real_template_keeps_tools_without_grammar():
    from runtime_chat_effort import ChatEffortTest
    ChatEffortTest.setUpClass()
    fixture=ChatEffortTest();fixture.setUp()
    chat=fixture.chat
    chat.tokenizer_manager.server_args.grammar_backend='none'
    chat.tokenizer_manager.model_config.hf_config.model_type='qwen3_8_flash_next'
    chat.tool_call_parser='qwen3_coder'
    req=ChatCompletionRequest(model='test',messages=[{'role':'user','content':'Search nginx'}],tools=[{'type':'function','function':{'name':'probe','strict':True,'parameters':{'type':'object','properties':{'query':{'type':'string'}}}}}])
    result=chat._process_messages(req,False)
    assert result.tool_call_constraint is None
    rendered=chat.tokenizer_manager.tokenizer.decode(result.prompt_ids)
    assert 'probe' in rendered and 'query' in rendered and 'nginx' in rendered
