import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type {
  Api,
  AssistantMessage,
  Context,
  Model,
  SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
  callVisionModel,
  delegateToVisionModel,
} from "../lib/delegate.ts";
import { DEFAULT_CONFIG } from "../lib/config.ts";
import { VisionCache } from "../lib/cache.ts";

const PNG_1x1_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M8AAAMBEg1+mP0AAAAASUVORK5CYII=";
const PNG_BYTES = Buffer.from(PNG_1x1_B64, "base64");

function makeVisionModel(overrides: Partial<Model<Api>> = {}): Model<Api> {
  return {
    id: "minimax-m3:cloud",
    name: "MiniMax M3",
    api: "openai-completions" as Api,
    provider: "ollama",
    baseUrl: "http://localhost:11434/v1",
    reasoning: false,
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 512000,
    maxTokens: 8192,
    ...overrides,
  };
}

interface TransportCall {
  model: Model<Api>;
  context: Context;
  options: SimpleStreamOptions | undefined;
}

interface MockProvider {
  calls: TransportCall[];
  transport: {
    streamSimple(
      model: Model<Api>,
      context: Context,
      options?: SimpleStreamOptions,
    ): any;
  };
  restore: () => void;
}

let activeTransport: MockProvider["transport"] | undefined;

function assistantMessage(
  model: Model<Api>,
  content: AssistantMessage["content"],
  stopReason: AssistantMessage["stopReason"],
  errorMessage?: string,
): AssistantMessage {
  return {
    role: "assistant",
    content,
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason,
    errorMessage,
    timestamp: Date.now(),
  };
}

function mockFetchSeq(responses: { status: number; body: unknown }[]): {
  calls: TransportCall[];
  transport: MockProvider["transport"];
  restore: () => void;
} {
  const calls: TransportCall[] = [];
  let i = 0;
  const transport: MockProvider["transport"] = {
    streamSimple(model, context, options) {
      calls.push({ model, context, options });
      const response = responses[Math.min(i, responses.length - 1)]!;
      i++;
      return {
        result: async () => {
          if (options?.signal?.aborted) {
            return assistantMessage(model, [], "aborted", "Aborted");
          }
          await options?.onResponse?.(
            { status: response.status, headers: {} },
            model,
          );
          if (response.status >= 400) {
            const detail = typeof response.body === "string"
              ? response.body
              : JSON.stringify(response.body);
            return assistantMessage(model, [], "error", detail);
          }
          const body = response.body as {
            choices?: Array<{
              message?: { content?: string; reasoning_content?: string };
            }>;
          };
          const message = body.choices?.[0]?.message;
          const content: AssistantMessage["content"] = message?.content
            ? [{ type: "text", text: message.content }]
            : message?.reasoning_content
              ? [{ type: "thinking", thinking: message.reasoning_content }]
              : [];
          return assistantMessage(model, content, "stop");
        },
      };
    },
  };
  activeTransport = transport;
  return {
    calls,
    transport,
    restore: () => {
      if (activeTransport === transport) activeTransport = undefined;
    },
  };
}

function mockFetch(response: { status: number; body: unknown }): MockProvider {
  return mockFetchSeq([response]);
}

function mockFetchError(status: number, body: string): MockProvider {
  return mockFetch({ status, body });
}

function makeCtx(opts: {
  model?: Model<Api> | undefined;
  authOk?: boolean;
  authError?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  cwd?: string;
  transport?: MockProvider["transport"];
}): ExtensionContext {
  const model = opts.model ?? makeVisionModel();
  return {
    cwd: opts.cwd ?? "/tmp",
    modelRegistry: {
      find: () => opts.model === null ? undefined : model,
      getProvider: () => opts.transport ?? activeTransport,
      getApiKeyAndHeaders: async () =>
        opts.authOk === false
          ? { ok: false, error: opts.authError ?? "no api key" }
          : { ok: true, apiKey: opts.apiKey ?? "test-key", headers: opts.headers },
    } as unknown as ExtensionContext["modelRegistry"],
  } as unknown as ExtensionContext;
}

function tmpDir(): string {
  return mkdtempSync(join(tmpdir(), "vision-delegate-"));
}

test("callVisionModel: sends normalized image context through Pi transport", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "a red square" } }] } });
  try {
    const model = makeVisionModel({ api: "openai-responses" as Api });
    const text = await callVisionModel(
      model,
      m.transport,
      { ok: true, apiKey: "key-123", baseUrl: "https://resolved.example/v1" },
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "describe this",
      undefined,
      "off",
    );
    assert.equal(text, "a red square");
    assert.equal(m.calls.length, 1);
    const call = m.calls[0]!;
    assert.equal(call.model.api, "openai-responses", "model API is preserved for provider dispatch");
    assert.equal(call.model.baseUrl, "https://resolved.example/v1", "resolved auth can override the endpoint");
    assert.equal(call.context.messages[0]!.role, "user");
    const content = call.context.messages[0]!.content;
    assert.ok(Array.isArray(content));
    assert.deepEqual(content.map((part) => part.type), ["image", "text"]);
    assert.deepEqual(content[0], { type: "image", data: PNG_1x1_B64, mimeType: "image/png" });
    assert.deepEqual(content[1], { type: "text", text: "describe this" });
    assert.equal(call.options?.apiKey, "key-123");
    assert.equal(call.options?.temperature, undefined, "provider uses its protocol-safe default");
    assert.equal(call.options?.maxTokens, 4096);
  } finally {
    m.restore();
  }
});

test("callVisionModel: falls back to reasoning_content when content is empty", async () => {
  const m = mockFetch({
    status: 200,
    body: { choices: [{ message: { content: "", reasoning_content: "thought: it is blue" } }] },
  });
  try {
    const text = await callVisionModel(
      makeVisionModel(),
      m.transport,
      { ok: true },
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "what color",
      undefined,
      "off",
    );
    assert.equal(text, "thought: it is blue");
  } finally {
    m.restore();
  }
});

test("callVisionModel: passes generic reasoning through Pi transport", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    await callVisionModel(
      makeVisionModel({ reasoning: true }),
      m.transport,
      { ok: true, apiKey: "k" },
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "p",
      undefined,
      "high",
    );
    assert.equal(m.calls[0]!.options?.reasoning, "high");
  } finally {
    m.restore();
  }
});

test("callVisionModel: omits generic reasoning when level is off", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    await callVisionModel(
      makeVisionModel({ reasoning: true }),
      m.transport,
      { ok: true, apiKey: "k" },
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "p",
      undefined,
      "off",
    );
    assert.equal(m.calls[0]!.options?.reasoning, undefined);
  } finally {
    m.restore();
  }
});

test("callVisionModel: forwards resolved provider auth", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    await callVisionModel(
      makeVisionModel(),
      m.transport,
      { ok: true, apiKey: "k", headers: { "X-Custom": "yes" }, env: { REGION: "test" } },
      { data: PNG_1x1_B64, mimeType: "image/png" },
      "p",
      undefined,
      "off",
    );
    assert.equal(m.calls[0]!.options?.apiKey, "k");
    assert.equal(m.calls[0]!.options?.headers?.["X-Custom"], "yes");
    assert.equal(m.calls[0]!.options?.env?.REGION, "test");
  } finally {
    m.restore();
  }
});

test("callVisionModel: provider error includes the observed HTTP status", async () => {
  const m = mockFetchError(500, "internal server boom");
  try {
    await assert.rejects(
      callVisionModel(
        makeVisionModel(),
        m.transport,
        { ok: true, apiKey: "k" },
        { data: PNG_1x1_B64, mimeType: "image/png" },
        "p",
        undefined,
        "off",
      ),
      /500: internal server boom/,
    );
  } finally {
    m.restore();
  }
});

test("callVisionModel: no content in response throws", async () => {
  const m = mockFetch({ status: 200, body: { choices: [{ message: {} }] } });
  try {
    await assert.rejects(
      callVisionModel(
        makeVisionModel(),
        m.transport,
        { ok: true, apiKey: "k" },
        { data: PNG_1x1_B64, mimeType: "image/png" },
        "p",
        undefined,
        "off",
      ),
      /no content/,
    );
  } finally {
    m.restore();
  }
});

test("delegateToVisionModel: reuses the registered Pi provider transport", async () => {
  const dir = tmpDir();
  const file = join(dir, "pixel.png");
  writeFileSync(file, PNG_BYTES);

  const model = makeVisionModel({
    api: "openai-responses" as Api,
    provider: "openai-third",
    id: "gpt-5.6-terra",
    baseUrl: "https://gateway.example/v1",
    reasoning: true,
  });
  let transportCall: {
    model: Model<Api>;
    context: any;
    options: any;
  } | undefined;
  const provider = {
    streamSimple(requestModel: Model<Api>, context: any, options: any) {
      transportCall = { model: requestModel, context, options };
      return {
        result: async () => ({
          role: "assistant",
          content: [{ type: "text", text: "described through Pi" }],
          api: requestModel.api,
          provider: requestModel.provider,
          model: requestModel.id,
          usage: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
          stopReason: "stop",
          timestamp: Date.now(),
        }),
      };
    },
  };
  const ctx = {
    cwd: dir,
    modelRegistry: {
      find: () => model,
      getProvider: () => provider,
      getApiKeyAndHeaders: async () => ({
        ok: true,
        apiKey: "provider-key",
        headers: { "X-Provider": "yes" },
      }),
    },
  } as unknown as ExtensionContext;

  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    throw new Error("delegate bypassed Pi provider transport");
  }) as typeof globalThis.fetch;
  try {
    const cfg = {
      ...DEFAULT_CONFIG,
      provider: "openai-third",
      model: "gpt-5.6-terra",
      systemPrompt: "Analyze visible UI precisely.",
      retryAttempts: 0,
    };
    const result = await delegateToVisionModel(ctx, cfg, {
      image_path: file,
      prompt: "read the screenshot",
      compress: false,
      reasoning: "high",
    }, undefined);

    assert.equal(result.ok, true);
    if (result.ok) assert.equal(result.text, "described through Pi");
    assert.ok(transportCall, "registered provider transport was called");
    assert.equal(transportCall.model.api, "openai-responses");
    assert.equal(transportCall.context.systemPrompt, "Analyze visible UI precisely.");
    assert.deepEqual(
      transportCall.context.messages[0].content.map((part: { type: string }) => part.type),
      ["image", "text"],
    );
    assert.equal(transportCall.options.apiKey, "provider-key");
    assert.equal(transportCall.options.headers["X-Provider"], "yes");
    assert.equal(transportCall.options.reasoning, "high");
  } finally {
    globalThis.fetch = originalFetch;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: Pi <0.84 falls back to the registered API transport", async () => {
  const dir = tmpDir();
  const file = join(dir, "pixel.png");
  writeFileSync(file, PNG_BYTES);
  const api = "vision-test-legacy-api" as Api;
  const sourceId = "getpipher-vision-legacy-transport-test";
  const model = makeVisionModel({ api, provider: "legacy-provider" });
  const mock = mockFetch({
    status: 200,
    body: { choices: [{ message: { content: "legacy transport response" } }] },
  });
  const { registerApiProvider, unregisterApiProviders } = await import("@earendil-works/pi-ai/compat");
  registerApiProvider({
    api,
    stream: mock.transport.streamSimple as any,
    streamSimple: mock.transport.streamSimple as any,
  }, sourceId);
  const ctx = {
    cwd: dir,
    modelRegistry: {
      find: () => model,
      getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "legacy-key" }),
      // Intentionally no getProvider(): this is the pre-0.84 registry shape.
    },
  } as unknown as ExtensionContext;

  try {
    const result = await delegateToVisionModel(ctx, {
      ...DEFAULT_CONFIG,
      provider: "legacy-provider",
      model: model.id,
      retryAttempts: 0,
    }, {
      image_path: file,
      prompt: "describe",
      compress: false,
      reasoning: "off",
    }, undefined);

    assert.equal(result.ok, true);
    if (result.ok) assert.equal(result.text, "legacy transport response");
    assert.equal(mock.calls.length, 1);
  } finally {
    unregisterApiProviders(sourceId);
    mock.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: disabled → actionable error", async () => {
  const ctx = makeCtx({});
  const r = await delegateToVisionModel(ctx, { ...DEFAULT_CONFIG, enabled: false }, {
    image_path: "/x.png", prompt: "p", compress: true, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "disabled");
    assert.match(r.error.message, /\/vision on/);
  }
});

test("delegateToVisionModel: not configured → actionable error with /vision config", async () => {
  const ctx = makeCtx({});
  const r = await delegateToVisionModel(ctx, DEFAULT_CONFIG, {
    image_path: "/x.png", prompt: "p", compress: true, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "not_configured");
    assert.match(r.error.message, /\/vision config provider/);
  }
});

test("delegateToVisionModel: model not found → actionable error", async () => {
  const ctx = makeCtx({ model: null as unknown as Model<Api> });
  const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "bogus" };
  const r = await delegateToVisionModel(ctx, cfg, {
    image_path: "/x.png", prompt: "p", compress: true, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "model_not_found");
    assert.match(r.error.message, /models\.json/);
  }
});

test("delegateToVisionModel: auth failure → actionable error", async () => {
  const ctx = makeCtx({ authOk: false, authError: "key not set" });
  const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud" };
  const r = await delegateToVisionModel(ctx, cfg, {
    image_path: "/x.png", prompt: "p", compress: false, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "auth_error");
    assert.match(r.error.message, /key not set/);
  }
});

test("delegateToVisionModel: image not found → actionable error with path", async () => {
  const ctx = makeCtx({});
  const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud" };
  const r = await delegateToVisionModel(ctx, cfg, {
    image_path: "/nonexistent-vision-test.png", prompt: "p", compress: false, reasoning: "off",
  }, undefined);
  assert.equal(r.ok, false);
  if (!r.ok) {
    assert.equal(r.error.code, "not_found");
    assert.match(r.error.message, /nonexistent-vision-test\.png/);
  }
});

test("delegateToVisionModel: success → returns text + details", async () => {
  const dir = tmpDir();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "a tiny image" } }] } });
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud" };
    const r = await delegateToVisionModel(ctx, cfg, {
      image_path: file, prompt: "describe", compress: false, reasoning: "off",
    }, undefined);
    assert.equal(r.ok, true);
    if (r.ok) {
      assert.equal(r.text, "a tiny image");
      assert.equal(r.details.model, "ollama/minimax-m3:cloud");
      assert.equal(r.details.image_path, file);
      assert.equal(r.details.compressed, false);
    }
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

// ── v0.2.0 (SPEC-2) tests ───────────────────────────────────────────────

test("delegateToVisionModel: success details include cached=false + fallback=false (v0.1.0 path)", async () => {
  const dir = tmpDir();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud" };
    const r = await delegateToVisionModel(ctx, cfg, {
      image_path: file, prompt: "describe", compress: false, reasoning: "off",
    }, undefined);
    assert.equal(r.ok, true);
    if (r.ok) {
      assert.equal(r.details.cached, false);
      assert.equal(r.details.fallback, false);
    }
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: cache hit = 0 vision-model calls (2nd call)", async () => {
  const dir = tmpDir();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "a desc" } }] } });
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", cacheEnabled: true };
    const cache = new VisionCache(undefined, 256);
    const params = { image_path: file, prompt: "describe", compress: false, reasoning: "off" as const };

    const r1 = await delegateToVisionModel(ctx, cfg, params, undefined, cache);
    assert.equal(r1.ok, true);
    if (r1.ok) assert.equal(r1.details.cached, false);
    assert.equal(m.calls.length, 1, "first call fetches");

    const r2 = await delegateToVisionModel(ctx, cfg, params, undefined, cache);
    assert.equal(r2.ok, true);
    if (r2.ok) {
      assert.equal(r2.details.cached, true, "second call is a cache hit");
      assert.equal(r2.text, "a desc");
    }
    assert.equal(m.calls.length, 1, "second call = 0 vision-model calls");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: cache miss when prompt changes", async () => {
  const dir = tmpDir();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "d" } }] } });
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", cacheEnabled: true };
    const cache = new VisionCache(undefined, 256);
    await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "describe", compress: false, reasoning: "off" }, undefined, cache);
    await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "different", compress: false, reasoning: "off" }, undefined, cache);
    assert.equal(m.calls.length, 2, "different prompt → miss → fetches again");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: systemPrompt is passed through normalized context", async () => {
  const dir = tmpDir();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", systemPrompt: "You are a forensic analyst." };
    await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(m.calls[0]!.context.systemPrompt, "You are a forensic analyst.");
    assert.equal(m.calls[0]!.context.messages[0]!.role, "user");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: no systemPrompt leaves normalized context unset", async () => {
  const dir = tmpDir();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud" };
    await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(m.calls[0]!.context.systemPrompt, undefined);
    assert.equal(m.calls[0]!.context.messages[0]!.role, "user");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: retry on 500 then success (retryAttempts=2 → 3 calls)", async () => {
  const dir = tmpDir();
  const m = mockFetchSeq([
    { status: 500, body: { error: "boom" } },
    { status: 500, body: { error: "boom" } },
    { status: 200, body: { choices: [{ message: { content: "recovered" } }] } },
  ]);
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", retryAttempts: 2, retryBackoffMs: 1 };
    const r = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(r.ok, true);
    if (r.ok) {
      assert.equal(r.text, "recovered");
      assert.equal(r.details.fallback, false);
      assert.equal(r.details.cached, false);
    }
    assert.equal(m.calls.length, 3, "3 total attempts on primary");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: 4xx (client) → no retry → fallback fires", async () => {
  const dir = tmpDir();
  const m = mockFetchSeq([
    { status: 400, body: "bad request" },        // primary (no retry)
    { status: 200, body: { choices: [{ message: { content: "fb-desc" } }] } }, // fallback
  ]);
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = {
      ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud",
      retryAttempts: 3, retryBackoffMs: 1,
      fallbackProvider: "openrouter", fallbackModel: "qwen3.5:cloud",
    };
    const r = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(r.ok, true);
    if (r.ok) {
      assert.equal(r.details.fallback, true);
      assert.equal(r.details.model, "openrouter/qwen3.5:cloud");
      assert.equal(r.text, "fb-desc");
    }
    assert.equal(m.calls.length, 2, "1 primary (no retry) + 1 fallback");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: 5xx exhausts retries → fallback fires", async () => {
  const dir = tmpDir();
  const m = mockFetchSeq([
    { status: 500, body: "boom" },
    { status: 500, body: "boom" },
    { status: 500, body: "boom" },              // primary exhausts (attempts=2 → 3 calls)
    { status: 200, body: { choices: [{ message: { content: "fb" } }] } }, // fallback
  ]);
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = {
      ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud",
      retryAttempts: 2, retryBackoffMs: 1,
      fallbackProvider: "openrouter", fallbackModel: "qwen3.5:cloud",
    };
    const r = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(r.ok, true);
    if (r.ok) assert.equal(r.details.fallback, true);
    assert.equal(m.calls.length, 4, "3 primary + 1 fallback");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: no fallback configured + primary exhausts → primary error", async () => {
  const dir = tmpDir();
  const m = mockFetchSeq([
    { status: 500, body: "boom" },
    { status: 500, body: "boom" },
    { status: 500, body: "boom" },
  ]);
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", retryAttempts: 2, retryBackoffMs: 1 };
    const r = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(r.ok, false);
    if (!r.ok) {
      assert.equal(r.error.code, "vision_call_error");
      assert.match(r.error.message, /500/);
      assert.equal(r.details, undefined, "no fallback attempted → no primaryError details");
    }
    assert.equal(m.calls.length, 3);
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: fallback also fails → error + primaryError + fallbackModel", async () => {
  const dir = tmpDir();
  const m = mockFetchSeq([
    { status: 500, body: "primary boom" },
    { status: 500, body: "primary boom" },
    { status: 500, body: "primary boom" },
    { status: 500, body: "fallback boom" },   // fallback also fails
  ]);
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const cfg = {
      ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud",
      retryAttempts: 2, retryBackoffMs: 1,
      fallbackProvider: "openrouter", fallbackModel: "qwen3.5:cloud",
    };
    const r = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(r.ok, false);
    if (!r.ok) {
      assert.equal(r.error.code, "vision_call_error");
      assert.match(r.error.message, /fallback/);
      assert.equal(r.details?.fallbackModel, "openrouter/qwen3.5:cloud");
      assert.ok(r.details?.primaryError, "primaryError set for traceability");
    }
    assert.equal(m.calls.length, 4);
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("delegateToVisionModel: abort → code 'aborted', 0 calls, no fallback", async () => {
  const dir = tmpDir();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "ok" } }] } });
  try {
    const file = join(dir, "pixel.png");
    writeFileSync(file, PNG_BYTES);
    const ctx = makeCtx({ cwd: dir });
    const ac = new AbortController();
    ac.abort();
    const cfg = {
      ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud",
      retryAttempts: 3, retryBackoffMs: 1,
      fallbackProvider: "openrouter", fallbackModel: "qwen3.5:cloud",
    };
    const r = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, ac.signal);
    assert.equal(r.ok, false);
    if (!r.ok) assert.equal(r.error.code, "aborted");
    assert.equal(m.calls.length, 0, "abort before first attempt → 0 calls + no fallback");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});
// ── v0.5.0 (SPEC-5) tests: audit log + local-only mode ───────────────────

import { readFileSync, existsSync, mkdirSync } from "node:fs";
import { countAuditLog, tailAuditLog, clearAuditLog } from "../lib/audit.ts";

/** Helper: a temp dir + an image file + a configured ctx + an audit log path. */
function setupAuditTest() {
  const dir = mkdtempSync(join(tmpdir(), "vision-delegate-audit-"));
  const file = join(dir, "pixel.png");
  writeFileSync(file, PNG_BYTES);
  const auditPath = join(dir, "audit.log");
  const ctx = makeCtx({ cwd: dir });
  const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", auditLog: true, auditLogPath: auditPath };
  return { dir, file, auditPath, ctx, cfg };
}

test("T54: audit log basic — success → one JSONL line with full routing trace, no bytes", async () => {
  const { dir, file, auditPath, ctx, cfg } = setupAuditTest();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "a desc" } }] } });
  try {
    const r = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "describe", compress: false, reasoning: "off" }, undefined);
    assert.equal(r.ok, true);
    assert.equal(countAuditLog(auditPath), 1, "one audit line");
    const lines = readFileSync(auditPath, "utf8").split("\n").filter((l) => l.trim().length > 0);
    const entry = JSON.parse(lines[0]!);
    assert.equal(entry.provider, "ollama");
    assert.equal(entry.model, "ollama/minimax-m3:cloud");
    assert.equal(entry.image_path, file, "file path logged in full");
    assert.equal(entry.cached, false);
    assert.equal(entry.fallback, false);
    assert.equal(entry.ok, true);
    assert.equal(entry.error_code, undefined);
    assert.equal(entry.local_only, false);
    assert.ok(entry.latency_ms >= 0, "latency measured");
    assert.ok(typeof entry.source_hash === "string" && entry.source_hash.length > 0, "source hash present");
    // Privacy: the raw image bytes must NOT appear in the log line.
    assert.ok(!lines[0]!.includes(PNG_1x1_B64), "no image bytes in audit log");
    assert.equal(m.calls.length, 1, "one fetch");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("T55: audit log cache hit → second line cached:true, ok:true, latency_ms:0, fetch not called", async () => {
  const { dir, file, auditPath, ctx, cfg } = setupAuditTest();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "a desc" } }] } });
  try {
    const cache = new VisionCache(undefined, 256);
    const params = { image_path: file, prompt: "describe", compress: false, reasoning: "off" as const };
    await delegateToVisionModel(ctx, cfg, params, undefined, cache); // miss → fetch
    await delegateToVisionModel(ctx, cfg, params, undefined, cache); // hit → no fetch
    assert.equal(countAuditLog(auditPath), 2, "two audit lines");
    const lines = readFileSync(auditPath, "utf8").split("\n").filter((l) => l.trim().length > 0);
    const hitEntry = JSON.parse(lines[1]!);
    assert.equal(hitEntry.cached, true);
    assert.equal(hitEntry.ok, true);
    assert.equal(hitEntry.latency_ms, 0, "cache hit = 0 latency");
    assert.equal(hitEntry.local_only, false);
    assert.equal(m.calls.length, 1, "second call = 0 vision-model calls");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("T56: audit log fallback success → fallback:true, fallback_model set; then both fail → error_code", async () => {
  const dir = mkdtempSync(join(tmpdir(), "vision-delegate-audit-"));
  const file = join(dir, "pixel.png");
  writeFileSync(file, PNG_BYTES);
  const auditPath = join(dir, "audit.log");
  const ctx = makeCtx({ cwd: dir });
  const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", fallbackProvider: "openrouter", fallbackModel: "gpt-4o", auditLog: true, auditLogPath: auditPath, retryAttempts: 0, retryBackoffMs: 1 };
  // Primary fails (500), fallback succeeds (200).
  const m = mockFetchSeq([
    { status: 500, body: { error: "boom" } },
    { status: 200, body: { choices: [{ message: { content: "fallback desc" } }] } },
  ]);
  try {
    const r1 = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(r1.ok, true, "fallback succeeds");
    if (r1.ok) assert.equal(r1.details.fallback, true);
    assert.equal(countAuditLog(auditPath), 1);
    const e1 = JSON.parse(readFileSync(auditPath, "utf8").split("\n").filter((l) => l.trim())[0]!);
    assert.equal(e1.fallback, true);
    assert.equal(e1.ok, true);
    assert.equal(e1.fallback_model, "openrouter/gpt-4o", "fallback model recorded");
    // provider = configured primary (the attempted route), per PLAN-5 §1.6.
    assert.equal(e1.provider, "ollama");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }

  // Now: both primary + fallback fail.
  const dir2 = mkdtempSync(join(tmpdir(), "vision-delegate-audit-"));
  const file2 = join(dir2, "pixel.png");
  writeFileSync(file2, PNG_BYTES);
  const auditPath2 = join(dir2, "audit.log");
  const ctx2 = makeCtx({ cwd: dir2 });
  const cfg2 = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", fallbackProvider: "openrouter", fallbackModel: "gpt-4o", auditLog: true, auditLogPath: auditPath2, retryAttempts: 0, retryBackoffMs: 1 };
  const m2 = mockFetchSeq([
    { status: 500, body: { error: "primary boom" } },
    { status: 500, body: { error: "fallback boom" } },
  ]);
  try {
    const r2 = await delegateToVisionModel(ctx2, cfg2, { image_path: file2, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(r2.ok, false, "both fail");
    assert.equal(countAuditLog(auditPath2), 1);
    const e2 = JSON.parse(readFileSync(auditPath2, "utf8").split("\n").filter((l) => l.trim())[0]!);
    assert.equal(e2.ok, false);
    assert.equal(e2.error_code, "vision_call_error");
  } finally {
    m2.restore();
    rmSync(dir2, { recursive: true, force: true });
  }
});

test("T57: local-only cache hit → returns cached desc, fetch NOT called, local_only:true", async () => {
  const { dir, file, auditPath, ctx, cfg } = setupAuditTest();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "cached desc" } }] } });
  try {
    const cache = new VisionCache(undefined, 256);
    const params = { image_path: file, prompt: "describe", compress: false, reasoning: "off" as const };
    // Prime the cache with localOnly OFF.
    const cfgNormal = { ...cfg, localOnly: false };
    await delegateToVisionModel(ctx, cfgNormal, params, undefined, cache);
    assert.equal(m.calls.length, 1, "primed cache with one fetch");
    // Now: localOnly ON, same image → cache hit, no new fetch.
    clearAuditLog(auditPath);
    const cfgLocal = { ...cfg, localOnly: true };
    const r = await delegateToVisionModel(ctx, cfgLocal, params, undefined, cache);
    assert.equal(r.ok, true, "cache hit returns desc");
    if (r.ok) assert.equal(r.text, "cached desc");
    assert.equal(m.calls.length, 1, "fetch NOT called again (cache hit in local-only)");
    // Audit: cached:true, local_only:true, ok:true.
    assert.equal(countAuditLog(auditPath), 1);
    const entry = JSON.parse(readFileSync(auditPath, "utf8").split("\n").filter((l) => l.trim())[0]!);
    assert.equal(entry.cached, true);
    assert.equal(entry.local_only, true);
    assert.equal(entry.ok, true);
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("T58: local-only cache miss → clear error, fetch NOT called (structural guarantee)", async () => {
  const { dir, file, auditPath, ctx, cfg } = setupAuditTest();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "should not reach" } }] } });
  try {
    const cfgLocal = { ...cfg, localOnly: true };
    const r = await delegateToVisionModel(ctx, cfgLocal, { image_path: file, prompt: "describe", compress: false, reasoning: "off" }, undefined);
    assert.equal(r.ok, false, "cache miss + local-only → refusal");
    if (!r.ok) {
      assert.equal(r.error.code, "local_only");
      assert.ok(r.error.message.includes("local-only mode"), "clear message");
      assert.ok(r.error.message.includes("/vision local-only off"), "actionable: names the toggle");
    }
    assert.equal(m.calls.length, 0, "fetch NOT called — structural guarantee (no network)");
    // Audit: ok:false, error_code:"local_only", local_only:true, latency_ms:0.
    assert.equal(countAuditLog(auditPath), 1);
    const entry = JSON.parse(readFileSync(auditPath, "utf8").split("\n").filter((l) => l.trim())[0]!);
    assert.equal(entry.ok, false);
    assert.equal(entry.error_code, "local_only");
    assert.equal(entry.local_only, true);
    assert.equal(entry.latency_ms, 0);
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("T66: audit disabled (auditLog:false) → no file I/O, delegation succeeds", async () => {
  const dir = mkdtempSync(join(tmpdir(), "vision-delegate-audit-"));
  const file = join(dir, "pixel.png");
  writeFileSync(file, PNG_BYTES);
  const auditPath = join(dir, "audit.log");
  const ctx = makeCtx({ cwd: dir });
  const cfg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", auditLog: false, auditLogPath: auditPath };
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "a desc" } }] } });
  try {
    const r = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, undefined);
    assert.equal(r.ok, true, "delegation still works with audit off");
    assert.equal(existsSync(auditPath), false, "no audit file created (no I/O)");
    assert.equal(m.calls.length, 1);
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("audit on abort → entry error_code:aborted", async () => {
  const { dir, file, auditPath, ctx, cfg } = setupAuditTest();
  const m = mockFetchError(500, "boom");
  try {
    const controller = new AbortController();
    controller.abort();
    const cfgRetry = { ...cfg, retryAttempts: 0, fallbackProvider: undefined, fallbackModel: undefined };
    const r = await delegateToVisionModel(ctx, cfgRetry, { image_path: file, prompt: "p", compress: false, reasoning: "off" }, controller.signal);
    assert.equal(r.ok, false);
    if (!r.ok) assert.equal(r.error.code, "aborted");
    assert.equal(countAuditLog(auditPath), 1);
    const entry = JSON.parse(readFileSync(auditPath, "utf8").split("\n").filter((l) => l.trim())[0]!);
    assert.equal(entry.ok, false);
    assert.equal(entry.error_code, "aborted");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("audit boundary: pre-flight errors (not_configured, model_not_found, auth, image-not-found) NOT audited", async () => {
  const dir = mkdtempSync(join(tmpdir(), "vision-delegate-audit-"));
  const auditPath = join(dir, "audit.log");
  try {
    // not_configured
    const ctx1 = makeCtx({ cwd: dir });
    const cfgNC = { ...DEFAULT_CONFIG, auditLog: true, auditLogPath: auditPath };
    await delegateToVisionModel(ctx1, cfgNC, { image_path: "/tmp/x.png", prompt: "p", compress: false, reasoning: "off" }, undefined);

    // model_not_found (registry returns undefined)
    const ctx2 = { cwd: dir, modelRegistry: { find: () => undefined, getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "k" }) } as unknown as ExtensionContext["modelRegistry"] } as unknown as ExtensionContext;
    const cfgMN = { ...DEFAULT_CONFIG, provider: "ollama", model: "nope", auditLog: true, auditLogPath: auditPath };
    await delegateToVisionModel(ctx2, cfgMN, { image_path: "/tmp/x.png", prompt: "p", compress: false, reasoning: "off" }, undefined);

    // auth failure
    const ctx3 = makeCtx({ cwd: dir, authOk: false });
    const cfgAuth = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", auditLog: true, auditLogPath: auditPath };
    await delegateToVisionModel(ctx3, cfgAuth, { image_path: "/tmp/x.png", prompt: "p", compress: false, reasoning: "off" }, undefined);

    // image not found
    const ctx4 = makeCtx({ cwd: dir });
    const cfgImg = { ...DEFAULT_CONFIG, provider: "ollama", model: "minimax-m3:cloud", auditLog: true, auditLogPath: auditPath };
    await delegateToVisionModel(ctx4, cfgImg, { image_path: "/tmp/does-not-exist.png", prompt: "p", compress: false, reasoning: "off" }, undefined);

    // None of these should have written an audit line (pre-flight, before image load / before routing).
    assert.equal(countAuditLog(auditPath), 0, "pre-flight errors are NOT routing events → not audited");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("T47 regression: single-image success with audit on → v0.4.0 behavior preserved + 1 audit line (additive)", async () => {
  const { dir, file, auditPath, ctx, cfg } = setupAuditTest();
  const m = mockFetch({ status: 200, body: { choices: [{ message: { content: "a desc" } }] } });
  try {
    const r = await delegateToVisionModel(ctx, cfg, { image_path: file, prompt: "describe", compress: false, reasoning: "off" }, undefined);
    assert.equal(r.ok, true);
    if (r.ok) {
      assert.equal(r.text, "a desc");
      assert.equal(r.details.cached, false);
      assert.equal(r.details.fallback, false);
      assert.equal(r.details.model, "ollama/minimax-m3:cloud");
    }
    assert.equal(countAuditLog(auditPath), 1, "exactly one audit line (additive)");
  } finally {
    m.restore();
    rmSync(dir, { recursive: true, force: true });
  }
});


