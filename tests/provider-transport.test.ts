import { test } from "node:test";
import assert from "node:assert/strict";
import type { Api, Model } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { openAIResponsesApi } from "@earendil-works/pi-ai/api/openai-responses.lazy";
import { callVisionModel } from "../lib/delegate.ts";
import { classifyError } from "../lib/resilience.ts";

const PNG_1X1_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M8AAAMBEg1+mP0AAAAASUVORK5CYII=";

function responsesModel(): Model<Api> {
  return {
    id: "gpt-5.6-terra",
    name: "GPT-5.6 Terra",
    api: "openai-responses" as Api,
    provider: "openai-third",
    baseUrl: "https://gateway.example/v1",
    reasoning: true,
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 272000,
    maxTokens: 32768,
  };
}

function successfulResponsesStream(): string {
  const events = [
    {
      type: "response.output_item.added",
      output_index: 0,
      item: {
        type: "message",
        id: "msg_vision",
        role: "assistant",
        status: "in_progress",
        content: [],
      },
    },
    {
      type: "response.output_text.delta",
      output_index: 0,
      content_index: 0,
      item_id: "msg_vision",
      delta: "visible UI",
      logprobs: [],
    },
    {
      type: "response.output_item.done",
      output_index: 0,
      item: {
        type: "message",
        id: "msg_vision",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "visible UI", annotations: [] }],
      },
    },
    {
      type: "response.completed",
      response: {
        id: "resp_vision",
        object: "response",
        created_at: 0,
        status: "completed",
        model: "gpt-5.6-terra",
        output: [],
        usage: {
          input_tokens: 1,
          input_tokens_details: { cached_tokens: 0 },
          output_tokens: 1,
          output_tokens_details: { reasoning_tokens: 0 },
          total_tokens: 2,
        },
      },
    },
  ];
  return `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`;
}

function successfulChatCompletionsStream(): string {
  const chunks = [
    {
      id: "chatcmpl_vision",
      object: "chat.completion.chunk",
      created: 0,
      model: "minimax-m3:cloud",
      choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
    },
    {
      id: "chatcmpl_vision",
      object: "chat.completion.chunk",
      created: 0,
      model: "minimax-m3:cloud",
      choices: [{ index: 0, delta: { content: "visible UI" }, finish_reason: null }],
    },
    {
      id: "chatcmpl_vision",
      object: "chat.completion.chunk",
      created: 0,
      model: "minimax-m3:cloud",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    },
  ];
  return `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`;
}

test("Pi openai-responses transport sends vision requests to /responses", async () => {
  const requests: Array<{ url: string; body: any }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const request = new Request(input, init);
    requests.push({
      url: request.url,
      body: JSON.parse(await request.clone().text()),
    });
    return new Response(successfulResponsesStream(), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }) as typeof globalThis.fetch;

  try {
    const text = await callVisionModel(
      responsesModel(),
      openAIResponsesApi(),
      { ok: true, apiKey: "test-key" },
      { data: PNG_1X1_B64, mimeType: "image/png" },
      "read the screenshot",
      undefined,
      "high",
    );

    assert.equal(text, "visible UI");
    assert.equal(requests.length, 1);
    assert.equal(requests[0]!.url, "https://gateway.example/v1/responses");
    assert.ok(!requests[0]!.url.includes("chat/completions"));

    const userInput = requests[0]!.body.input.find(
      (item: { role?: string }) => item.role === "user",
    );
    assert.ok(userInput, "Responses payload contains a user input item");
    assert.deepEqual(
      userInput.content.map((part: { type: string }) => part.type),
      ["input_image", "input_text"],
    );
    assert.ok(userInput.content[0].image_url.startsWith("data:image/png;base64,"));
    assert.equal(userInput.content[1].text, "read the screenshot");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Pi openai-completions transport preserves Chat Completions compatibility", async () => {
  const requests: Array<{ url: string; body: any }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const request = new Request(input, init);
    requests.push({
      url: request.url,
      body: JSON.parse(await request.clone().text()),
    });
    return new Response(successfulChatCompletionsStream(), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }) as typeof globalThis.fetch;

  try {
    const model: Model<Api> = {
      ...responsesModel(),
      id: "minimax-m3:cloud",
      name: "MiniMax M3",
      api: "openai-completions" as Api,
      provider: "ollama",
      baseUrl: "https://ollama.example/v1",
      reasoning: false,
    };
    const text = await callVisionModel(
      model,
      openAICompletionsApi(),
      { ok: true, apiKey: "test-key" },
      { data: PNG_1X1_B64, mimeType: "image/png" },
      "read the screenshot",
      undefined,
      "off",
    );

    assert.equal(text, "visible UI");
    assert.equal(requests.length, 1);
    assert.equal(requests[0]!.url, "https://ollama.example/v1/chat/completions");
    assert.deepEqual(
      requests[0]!.body.messages[0].content.map((part: { type: string }) => part.type),
      ["image_url", "text"],
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Pi openai-responses 503 remains retryable for vision resilience", async () => {
  const originalFetch = globalThis.fetch;
  let requests = 0;
  globalThis.fetch = (async () => {
    requests++;
    return new Response(JSON.stringify({
      error: {
        message: "No available providers",
        type: "no_available_providers",
        code: "no_available_providers",
      },
    }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof globalThis.fetch;

  try {
    await assert.rejects(
      callVisionModel(
        responsesModel(),
        openAIResponsesApi(),
        { ok: true, apiKey: "test-key" },
        { data: PNG_1X1_B64, mimeType: "image/png" },
        "read the screenshot",
        undefined,
        "high",
      ),
      (error: unknown) => {
        assert.ok(error instanceof Error);
        assert.match(error.message, /503/);
        assert.match(error.message, /No available providers/);
        assert.equal(classifyError(error), "retryable");
        return true;
      },
    );
    assert.equal(requests, 1, "the provider adapter itself does not hide extra retries");
  } finally {
    globalThis.fetch = originalFetch;
  }
});