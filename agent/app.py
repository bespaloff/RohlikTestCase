#!/usr/bin/env python

import os
import logging
import logging.config
from typing import Any
from uuid import uuid4, UUID
import json

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import RunnableConfig
from pydantic import BaseModel
from langchain_core.messages import AIMessage

load_dotenv()

# There are tools set here dependent on environment variables
from graph import graph, weak_model # noqa

FOLLOWUP_QUESTION_NUMBER = 3
TRIM_MESSAGE_LENGTH = 16  # Includes tool messages
USER_INPUT_MAX_LENGTH = 10000  # Characters

# We need the same secret for data persistance
# If you store sensitive data, you should store your secret in .env
BROWSER_STORAGE_SECRET = "itsnosecret"

with open('logging-config.json', 'r') as fh:
    config = json.load(fh)
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

async def chat_fn(user_input: str, history: dict, input_graph_state: dict, uuid: UUID, prompt: str):
    """
    Args:
        user_input (str): The user's input message
        history (dict): The history of the conversation in gradio
        input_graph_state (dict): The current state of the graph. This includes tool call history
        uuid (UUID): The unique identifier for the current conversation. This can be used in conjunction with langgraph or for memory
        prompt (str): The system prompt
    Yields:
        str: The output message
        dict|Any: The final state of the graph
        bool|Any: Whether to trigger follow up questions

        We do not use gradio history in the graph since we want the ToolMessage in the history
        ordered properly. GraphProcessingState.messages is used as history instead
    """
    try:
        logger.info(f"Prompt: {prompt}")
        logger.info(f"User input: {user_input}")
        input_graph_state["tools_enabled"] = {
            "recipe_finder": True,
            "shopping_list_manager": True,
        }
        if prompt:
            input_graph_state["prompt"] = prompt
        if "messages" not in input_graph_state:
            input_graph_state["messages"] = []
        input_graph_state["messages"].append(
            HumanMessage(user_input[:USER_INPUT_MAX_LENGTH])
        )
        input_graph_state["messages"] = input_graph_state["messages"][-TRIM_MESSAGE_LENGTH:]
        config = RunnableConfig(
            recursion_limit=20,
            run_name="user_chat",
            configurable={"thread_id": uuid}
        )

        output: str = ""
        final_state: dict | Any = {}
        waiting_output_seq: list[str] = []

        async for item in graph.astream(
            input_graph_state,
            config=config,
            stream_mode=["values", "messages"],
            subgraphs=True,  # keep this on to see nested agent tokens
        ):
            # Normalize tuple shapes:
            # - (mode, chunk) when subgraphs=False
            # - (namespace, mode, chunk) when subgraphs=True
            if isinstance(item, tuple) and len(item) == 3:
                namespace, mode, chunk = item
            else:
                mode, chunk = item
                namespace = ()  # top-level

            if mode == "values":
                # Only trust top-level state as the "final" state
                if not namespace:
                    final_state = chunk
                    # Optional: surface tool activity hints from the LAST message
                    last = chunk.get("messages", [])[-1] if chunk.get("messages") else None
                    if last and getattr(last, "tool_calls", None):
                        for call in last.tool_calls:
                            tname = call.get("name")
                            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {}) or {}
                            if tname == "recipe_finder":
                                waiting_output_seq.append(f"🔍 Searching for recipes ({args.get('dietary_goal', '')})...")
                            elif tname == "shopping_list_manager":
                                waiting_output_seq.append(f"🛒 Managing shopping list ({args.get('action', '')})...")
                            else:
                                waiting_output_seq.append(f"Running {tname}...")
                        if waiting_output_seq:
                            yield "\n".join(waiting_output_seq), gr.skip(), gr.skip()

            elif mode == "messages":
                # messages stream yields (msg, metadata)
                msg, metadata = chunk
                if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                    output += msg.content
                    yield output, gr.skip(), gr.skip()

        # --- after the loop, guarantee something shows in UI ---
        if not output.strip():
            # Fallback to last AI message from final_state
            last_ai = next(
                (m for m in reversed(final_state.get("messages", []))
                if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()),
                None
            )
            if last_ai:
                output = last_ai.content

        yield output + " ", dict(final_state), True
    except Exception:
        logger.exception("Exception occurred")
        user_error_message = "There was an error processing your request. Please try again."
        yield user_error_message, gr.skip(), False

def clear():
    return dict(), uuid4()

class FollowupQuestions(BaseModel):
    """Model for langchain to use for structured output for followup questions"""
    questions: list[str]

async def populate_followup_questions(end_of_chat_response: bool, messages: dict[str, str], uuid: UUID):
    """
    This function gets called a lot due to the asynchronous nature of streaming

    Only populate followup questions if streaming has completed and the message is coming from the assistant
    """
    if not end_of_chat_response or not messages or messages[-1]["role"] != "assistant":
        return *[gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)], False
    config = RunnableConfig(
        run_name="populate_followup_questions",
        configurable={"thread_id": uuid}
    )
    weak_model_with_config = weak_model.with_config(config)
    follow_up_questions = await weak_model_with_config.with_structured_output(FollowupQuestions).ainvoke([
        ("system", f"suggest {FOLLOWUP_QUESTION_NUMBER} followup questions for the user to ask the assistant. Refrain from asking personal questions."),
        *messages,
    ])
    if len(follow_up_questions.questions) != FOLLOWUP_QUESTION_NUMBER:
        raise ValueError("Invalid value of followup questions")
    buttons = []
    for i in range(FOLLOWUP_QUESTION_NUMBER):
        buttons.append(
            gr.Button(follow_up_questions.questions[i], visible=True, elem_classes="chat-tab"),
        )
    return *buttons, False

async def summarize_chat(end_of_chat_response: bool, messages: dict, sidebar_summaries: dict, uuid: UUID):
    """Summarize chat for tab names"""
    # print("\n------------------------")
    # print("not end_of_chat_response", not end_of_chat_response)
    # print("not messages", not messages)
    # if messages:
    #     print("messages[-1][role] != assistant", messages[-1]["role"] != "assistant")
    # print("isinstance(sidebar_summaries, type(lambda x: x))", isinstance(sidebar_summaries, type(lambda x: x)))
    # print("uuid in sidebar_summaries", uuid in sidebar_summaries)
    should_return = (
        not end_of_chat_response or
        not messages or
        messages[-1]["role"] != "assistant" or
        # This is a bug with gradio
        isinstance(sidebar_summaries, type(lambda x: x)) or
        # Already created summary
        uuid in sidebar_summaries
    )
    if should_return:
        return gr.skip(), gr.skip()
    config = RunnableConfig(
        run_name="summarize_chat",
        configurable={"thread_id": uuid}
    )
    weak_model_with_config = weak_model.with_config(config)
    summary_response = await weak_model_with_config.ainvoke([
        ("system", "summarize this chat in 7 tokens or less. Refrain from using periods"),
        *messages,
    ])
    if uuid not in sidebar_summaries:
        sidebar_summaries[uuid] = summary_response.content
    return sidebar_summaries, False

async def new_tab(uuid, gradio_graph, messages, tabs, prompt, sidebar_summaries):
    new_uuid = uuid4()
    new_graph = {}
    if uuid not in sidebar_summaries:
        sidebar_summaries, _ = await summarize_chat(True, messages, sidebar_summaries, uuid)
    tabs[uuid] = {
        "graph": gradio_graph,
        "messages": messages,
        "prompt": prompt,
    }
    suggestion_buttons = []
    for _ in range(FOLLOWUP_QUESTION_NUMBER):
        suggestion_buttons.append(gr.Button(visible=False))
    new_messages = {}
    new_prompt = "You are a helpful assistant."
    return new_uuid, new_graph, new_messages, tabs, new_prompt, sidebar_summaries, *suggestion_buttons

def switch_tab(selected_uuid, tabs, gradio_graph, uuid, messages, prompt):
    # I don't know of another way to lookup uuid other than
    # by the button value

    # Save current state
    if messages:
        tabs[uuid] = {
            "graph": gradio_graph,
            "messages": messages,
            "prompt": prompt
        }

    if selected_uuid not in tabs:
        logger.error(f"Could not find the selected tab in offloaded_tabs_data_storage {selected_uuid}")
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    selected_tab_state = tabs[selected_uuid]
    selected_graph = selected_tab_state["graph"]
    selected_messages = selected_tab_state["messages"]
    selected_prompt = selected_tab_state.get("prompt", "")
    suggestion_buttons = []
    for _ in range(FOLLOWUP_QUESTION_NUMBER):
        suggestion_buttons.append(gr.Button(visible=False))
    return selected_graph, selected_uuid, selected_messages, tabs, selected_prompt, *suggestion_buttons

def delete_tab(current_chat_uuid, selected_uuid, sidebar_summaries, tabs):
    output_messages = gr.skip()
    if current_chat_uuid == selected_uuid:
        output_messages = dict()
    if selected_uuid in tabs:
        del tabs[selected_uuid]
    if selected_uuid in sidebar_summaries:
        del sidebar_summaries[selected_uuid]
    return sidebar_summaries, tabs, output_messages

def submit_edit_tab(selected_uuid, sidebar_summaries, text):
    sidebar_summaries[selected_uuid] = text
    return sidebar_summaries, ""

CSS = """
/* Base / reset */
footer { visibility: hidden }
.followup-question-button { font-size: 12px }

/* Use the full viewport height */
#app-root.gradio-container { height: 100vh !important; }

/* Dark theme polish */
body, #app-root { background: #0b0d10; color: #e5e7eb; }

/* Sidebar styling */
#left-sidebar { background: #0e1116; border-right: 1px solid #1f2937; }
#left-sidebar .chat-tab { font-size: 12px; padding-inline: 6px; background: #111827; color: #d1d5db; border: 1px solid #1f2937; }
#left-sidebar .chat-tab:hover { background: #172036; }
#left-sidebar .chat-tab.active { background: #24324d; color: #fff; border-color: #2b3e66; }
#new-chat-button { background-color: #1f2937; color: #e5e7eb; border: 1px solid #374151; }
#new-chat-button:hover { background-color: #2b3a54; }

.tab-button-control {
    min-width: 0;
    padding-left: 0;
    padding-right: 0;
}

/* Chat area should stretch and scroll internally */
#main-chatbot { height: calc(100vh - 260px) !important; }
#main-chatbot .wrap { height: 100% !important; overflow-y: auto; }

/* Chat bubble colors */
#main-chatbot .message.user { background: #1f2937; border-color: #374151; color: #e5e7eb; }
#main-chatbot .message.bot { background: #0f172a; border-color: #1f2937; color: #e5e7eb; }

/* Input area */
#chat-textbox textarea { background: #0e1116; color: #f3f4f6; border: 1px solid #1f2937; }
#chat-textbox textarea:focus { outline: none; box-shadow: 0 0 0 2px #2b3e66 inset; }
"""

# We set the ChatInterface textbox id to chat-textbox for this to work
TRIGGER_CHATINTERFACE_BUTTON = """
function triggerChatButtonClick() {

  // Find the div with id "chat-textbox"
  const chatTextbox = document.getElementById("chat-textbox");

  if (!chatTextbox) {
    console.error("Error: Could not find element with id 'chat-textbox'");
    return;
  }

  // Find the button that is a descendant of the div
  const button = chatTextbox.querySelector("button");

  if (!button) {
    console.error("Error: No button found inside the chat-textbox element");
    return;
  }

  // Trigger the click event
  button.click();
}"""

if __name__ == "__main__":
    logger.info("Starting the interface")
    with gr.Blocks(title="Meal Planning Assistant", fill_height=True, css=CSS, elem_id="app-root") as app:
        current_prompt_state = gr.BrowserState(
            storage_key="current_prompt_state",
            secret=BROWSER_STORAGE_SECRET,
        )
        current_uuid_state = gr.BrowserState(
            uuid4,
            storage_key="current_uuid_state",
            secret=BROWSER_STORAGE_SECRET,
        )
        current_langgraph_state = gr.BrowserState(
            dict(),
            storage_key="current_langgraph_state",
            secret=BROWSER_STORAGE_SECRET,
        )
        end_of_assistant_response_state = gr.State(
            bool(),
        )
        # [uuid] -> summary of chat
        sidebar_names_state = gr.BrowserState(
            dict(),
            storage_key="sidebar_names_state",
            secret=BROWSER_STORAGE_SECRET,
        )
        # [uuid] -> {"graph": gradio_graph, "messages": messages}
        offloaded_tabs_data_storage = gr.BrowserState(
            dict(),
            storage_key="offloaded_tabs_data_storage",
            secret=BROWSER_STORAGE_SECRET,
        )

        chatbot_message_storage = gr.BrowserState(
            [],
            storage_key="chatbot_message_storage",
            secret=BROWSER_STORAGE_SECRET,
        )
        # Hidden system prompt (kept above for wiring, not visually shown)
        prompt_textbox = gr.Textbox(
            value="You are a helpful meal planning assistant. Help users plan their meals and create shopping lists based on their dietary preferences.",
            label="System Prompt",
            interactive=True,
            visible=False
        )
        chatbot = gr.Chatbot(
            type="messages",
            scale=1,
            show_copy_button=True,
            height=600,
            editable="all",
            elem_id="main-chatbot",
        )
        tab_edit_uuid_state = gr.State(
            str()
        )
        prompt_textbox.change(lambda prompt: prompt, inputs=[prompt_textbox], outputs=[current_prompt_state])
        with gr.Sidebar(elem_id="left-sidebar") as sidebar:
            @gr.render(inputs=[tab_edit_uuid_state, end_of_assistant_response_state, sidebar_names_state, current_uuid_state, chatbot, offloaded_tabs_data_storage])
            def render_chats(tab_uuid_edit, end_of_chat_response, sidebar_summaries, active_uuid, messages, tabs):
                current_tab_button_text = ""
                if active_uuid not in sidebar_summaries:
                    current_tab_button_text = "Current Chat"
                elif active_uuid not in tabs:
                    current_tab_button_text = sidebar_summaries[active_uuid]
                if current_tab_button_text:
                    gr.Button(current_tab_button_text, elem_classes=["chat-tab", "active"])
                for chat_uuid, tab in reversed(tabs.items()):
                    elem_classes = ["chat-tab"]
                    if chat_uuid == active_uuid:
                        elem_classes.append("active")
                    button_uuid_state = gr.State(chat_uuid)
                    with gr.Row():
                        clear_tab_button = gr.Button(
                            "🗑",
                            scale=0,
                            elem_classes=["tab-button-control"]
                        )
                        clear_tab_button.click(
                            fn=delete_tab,
                            inputs=[
                                current_uuid_state,
                                button_uuid_state,
                                sidebar_names_state,
                                offloaded_tabs_data_storage
                            ],
                            outputs=[
                                sidebar_names_state,
                                offloaded_tabs_data_storage,
                                chat_interface.chatbot_value
                            ]
                        )
                        chat_button_text = sidebar_summaries.get(chat_uuid)
                        if not chat_button_text:
                            chat_button_text = str(chat_uuid)
                        if chat_uuid != tab_uuid_edit:
                            set_edit_tab_button = gr.Button(
                                "✎",
                                scale=0,
                                elem_classes=["tab-button-control"]
                            )
                            set_edit_tab_button.click(
                                fn=lambda x: x,
                                inputs=[button_uuid_state],
                                outputs=[tab_edit_uuid_state]
                            )
                            chat_tab_button = gr.Button(
                                chat_button_text,
                                elem_id=f"chat-{chat_uuid}-button",
                                elem_classes=elem_classes,
                                scale=2
                            )
                            chat_tab_button.click(
                                fn=switch_tab,
                                inputs=[
                                    button_uuid_state,
                                    offloaded_tabs_data_storage,
                                    current_langgraph_state,
                                    current_uuid_state,
                                    chatbot,
                                    prompt_textbox
                                ],
                                outputs=[
                                    current_langgraph_state,
                                    current_uuid_state,
                                    chat_interface.chatbot_value,
                                    offloaded_tabs_data_storage,
                                    prompt_textbox,
                                    *followup_question_buttons
                                ]
                            )
                        else:
                            chat_tab_text = gr.Textbox(
                                chat_button_text,
                                scale=2,
                                interactive=True,
                                show_label=False
                            )
                            chat_tab_text.submit(
                                fn=submit_edit_tab,
                                inputs=[
                                    button_uuid_state,
                                    sidebar_names_state,
                                    chat_tab_text
                                ],
                                outputs=[
                                    sidebar_names_state,
                                    tab_edit_uuid_state
                                ]
                            )
                    # )
                # return chat_tabs, sidebar_summaries
            new_chat_button = gr.Button("New Chat", elem_id="new-chat-button")
        chatbot.clear(fn=clear, outputs=[current_langgraph_state, current_uuid_state])
        with gr.Row():
            followup_question_buttons = []
            for i in range(FOLLOWUP_QUESTION_NUMBER):
                btn = gr.Button(f"Button {i+1}", visible=False)
                followup_question_buttons.append(btn)

        multimodal = False
        textbox_component = (
            gr.MultimodalTextbox if multimodal else gr.Textbox
        )
        with gr.Column():
            textbox = textbox_component(
                show_label=False,
                label="Message",
                placeholder="Tell me about your dietary preferences and how many days you'd like meal plans for...",
                scale=7,
                autofocus=True,
                submit_btn=True,
                stop_btn=True,
                elem_id="chat-textbox",
                lines=1,
            )
        chat_interface = gr.ChatInterface(
            chatbot=chatbot,
            fn=chat_fn,
            additional_inputs=[
                current_langgraph_state,
                current_uuid_state,
                prompt_textbox,
            ],
            additional_outputs=[
                current_langgraph_state,
                end_of_assistant_response_state
            ],
            type="messages",
            multimodal=multimodal,
            textbox=textbox,
        )

        # # Place title and informational text below the chat
        # with gr.Column():
        #     gr.Markdown("# 🍽️ AI Meal Planning Assistant")
        #     gr.Markdown("Tell me your dietary preferences and how many days you'd like to plan meals for. I'll create a personalized meal plan with a shopping list!")
        #     gr.Markdown("**Example requests:**")
        #     gr.Markdown("- *'I need a vegetarian meal plan for 3 days'*")
        #     gr.Markdown("- *'Plan low-carb meals for the next 5 days'*")
        #     gr.Markdown("- *'Help me with high-protein meals for a week'*")

        new_chat_button.click(
            new_tab,
            inputs=[
                current_uuid_state,
                current_langgraph_state,
                chatbot,
                offloaded_tabs_data_storage,
                prompt_textbox,
                sidebar_names_state,
            ],
            outputs=[
                current_uuid_state,
                current_langgraph_state,
                chat_interface.chatbot_value,
                offloaded_tabs_data_storage,
                prompt_textbox,
                sidebar_names_state,
                *followup_question_buttons,
            ]
        )

        def click_followup_button(btn):
            buttons = [gr.Button(visible=False) for _ in range(len(followup_question_buttons))]
            return btn, *buttons
        for btn in followup_question_buttons:
            btn.click(
                fn=click_followup_button,
                inputs=[btn],
                outputs=[
                    chat_interface.textbox,
                    *followup_question_buttons
                ]
            ).success(lambda: None, js=TRIGGER_CHATINTERFACE_BUTTON)

        chatbot.change(
            fn=populate_followup_questions,
            inputs=[
                end_of_assistant_response_state,
                chatbot,
                current_uuid_state
            ],
            outputs=[
                *followup_question_buttons,
                end_of_assistant_response_state
            ],
            trigger_mode="multiple"
        )
        chatbot.change(
            fn=summarize_chat,
            inputs=[
                end_of_assistant_response_state,
                chatbot,
                sidebar_names_state,
                current_uuid_state
            ],
            outputs=[
                sidebar_names_state,
                end_of_assistant_response_state
            ],
            trigger_mode="multiple"
        )
        chatbot.change(
            fn=lambda x: x,
            inputs=[chatbot],
            outputs=[chatbot_message_storage],
            trigger_mode="always_last"
        )

        @app.load(inputs=[chatbot_message_storage], outputs=[chat_interface.chatbot_value])
        def load_messages(messages):
            return messages

        @app.load(inputs=[current_prompt_state], outputs=[prompt_textbox])
        def load_prompt(current_prompt):
            return current_prompt

    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        # favicon_path="assets/favicon.ico"
    )
