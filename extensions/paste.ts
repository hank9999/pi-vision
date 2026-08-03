/**
 * @getpipher/vision — paste extension (SPEC-3: graduated from B-lite).
 *
 * Input event hook that makes pasted/referenced images visible + actionable
 * end-to-end, capability-aware for both multimodal AND text-only primaries.
 *
 * **Multimodal primary** (PASS-THROUGH): attaches images as native
 * attachments (B-lite delivery guarantee) + renders `[Image-#N]` markers in
 * the text (gap #6). Zero delegation — the model sees the image natively.
 *
 * **Text-only primary** (capability-aware branching, gap #8): does NOT
 * attach images (text-only models can't process them). Renders markers +
 * branches on `textOnlyPasteMode`:
 * - `"hint"` (default): markers + a hint line nudging the model to call
 *   `describe_image`. Zero tokens — the model decides.
 * - `"auto"` (opt-in): auto-delegates each image via the v0.2.x pipeline
 *   (cache/retry/fallback) + appends descriptions. Timeout-protected (own
 *   AbortController); falls back to hint on timeout/failure.
 * - `"off"`: markers only — no attachment, no hint, no delegation.
 *
 * Coexistence with pi-paster: pi-paster matches its own `[#image N]`
 * placeholders (different trigger); this hook matches file paths. Transforms
 * chain across handlers, and we dedup by data hash against `event.images`.
 *
 * Config + cache are shared via `lib/state.ts` (set by vision.ts on
 * session_start + every mutation).
 */
import { existsSync, statSync } from "node:fs";
import { isAbsolute, resolve as resolvePath } from "node:path";
import {
  CustomEditor,
  type ExtensionAPI,
  type ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import type { ImageContent } from "@earendil-works/pi-ai";
import { isKeyRelease, matchesKey, type EditorComponent } from "@earendil-works/pi-tui";
import { isMultimodal } from "../lib/capability.ts";
import { loadImage } from "../lib/image.ts";
import { renderMarkers, buildHintLine, buildDescriptionsBlock, styleMarker } from "../lib/marker.ts";
import { getSharedConfig, getSharedCache } from "../lib/state.ts";
import { delegateToVisionModel, type DelegateParams } from "../lib/delegate.ts";
import type { ReasoningLevel } from "../lib/config.ts";
import { createComposePreviewComponent, makePreviewImage } from "../lib/preview.ts";
import { mapWithConcurrency } from "../lib/batch.ts";

type EditorFactory = NonNullable<ReturnType<ExtensionContext["ui"]["getEditorComponent"]>>;

const IMAGE_EXT_RE = /\.(png|jpe?g|gif|webp|bmp)$/i;
const PI_DEFAULT_IMAGE_PASTE_INPUT = "\x16";
const IMAGE_MARKER_RE = /\[Image-#(\d+)\]/g;
const CLIPBOARD_IMAGE_PATH_RE = /(?:^|[\\/])pi-clipboard-[a-z0-9-]+\.(?:png|jpe?g|gif|webp)$/i;
// Matches path-like tokens (absolute /…, home ~/…, relative ./…/…/) ending in a
// known image extension. Allows \ (escaped space) which is what terminal
// drag-and-drop produces on paths with spaces (common on macOS screenshots).
// Unescaped-space paths are handled by the post-regex extension step below.
const PATH_TOKEN_RE = /(?:\/|~\/|\.{1,2}\/)(?:\\ |[^\s)"'<>])+\.(?:png|jpe?g|gif|webp|bmp)/gi;
const WINDOWS_PATH_TOKEN_RE = /[a-z]:\\[^\r\n)"'<>]*?\.(?:png|jpe?g|gif|webp|bmp)/gi;

/**
 * On macOS, expose Option+V as an alias for pi's built-in image paste action.
 * Pi owns clipboard access and inserts a /tmp/pi-clipboard-* path; the input
 * hook below then replaces that path with the configured [Image-#N] marker.
 */
export function remapMacOptionVPaste(
  data: string,
  platform: NodeJS.Platform = process.platform,
): { data: string } | undefined {
  if (platform !== "darwin" || isKeyRelease(data) || !matchesKey(data, "alt+v")) return undefined;
  return { data: PI_DEFAULT_IMAGE_PASTE_INPUT };
}

/** Tracks the real temp paths hidden behind compose-time image markers. */
export class ComposeImageReferences {
  private readonly pathsByIndex = new Map<number, string>();
  private readonly indexesByPath = new Map<string, number>();
  private nextIndex = 1;

  collapse(text: string): string {
    return this.collapseWithContext(text, text);
  }

  collapseInsertion(text: string, editorText: string): string {
    return this.collapseWithContext(text, editorText);
  }

  private collapseWithContext(text: string, editorText: string): string {
    let result = text;
    IMAGE_MARKER_RE.lastIndex = 0;
    const usedIndexes = new Set(
      [...`${editorText}\n${text}`.matchAll(IMAGE_MARKER_RE)].map((match) => Number(match[1])),
    );
    IMAGE_MARKER_RE.lastIndex = 0;

    for (const [index, path] of this.pathsByIndex) {
      if (!usedIndexes.has(index)) {
        this.pathsByIndex.delete(index);
        this.indexesByPath.delete(path);
      }
    }
    this.nextIndex = 1;

    const tokens = findImagePathTokens(text);
    const exactToken = text.trim();
    if (CLIPBOARD_IMAGE_PATH_RE.test(exactToken) && !tokens.includes(exactToken)) {
      tokens.push(exactToken);
    }

    for (const token of tokens) {
      const path = token.replace(/\\ /g, " ");
      if (!CLIPBOARD_IMAGE_PATH_RE.test(path)) continue;

      let index = this.indexesByPath.get(path);
      if (index === undefined) {
        while (usedIndexes.has(this.nextIndex) || this.pathsByIndex.has(this.nextIndex)) {
          this.nextIndex++;
        }
        index = this.nextIndex++;
        this.indexesByPath.set(path, index);
        this.pathsByIndex.set(index, path);
      }

      usedIndexes.add(index);
      result = result.split(token).join(`[Image-#${index}]`);
    }

    return result;
  }

  expand(text: string): string {
    IMAGE_MARKER_RE.lastIndex = 0;
    return text.replace(IMAGE_MARKER_RE, (marker, rawIndex: string, offset: number, source: string) => {
      const path = this.pathsByIndex.get(Number(rawIndex));
      if (!path) return marker;
      const nextCharacter = source[offset + marker.length];
      return nextCharacter === "[" ? `${path} ` : path;
    });
  }

  restorePaths(
    text: string,
    renderMarker: (index: number) => string,
    firstIndex = 1,
  ): string {
    let result = text;
    let index = firstIndex;
    for (const path of this.pathsByIndex.values()) {
      if (!result.includes(path)) continue;
      result = result.split(path).join(renderMarker(index));
      index++;
    }
    return result;
  }

  reset(): void {
    this.pathsByIndex.clear();
    this.indexesByPath.clear();
    this.nextIndex = 1;
  }
}

export function addComposeImageMarkersToEditor(
  editor: EditorComponent,
  references: ComposeImageReferences,
  colorMarker: (marker: string) => string = (marker) => marker,
): EditorComponent {
  const insertTextAtCursor = editor.insertTextAtCursor?.bind(editor);
  if (insertTextAtCursor) {
    editor.insertTextAtCursor = (text: string) => {
      insertTextAtCursor(references.collapseInsertion(text, editor.getText()));
    };
  }

  const render = editor.render.bind(editor);
  editor.render = (width: number) => render(width).map((line) => {
    IMAGE_MARKER_RE.lastIndex = 0;
    return line.replace(IMAGE_MARKER_RE, (marker) => colorMarker(marker));
  });
  return editor;
}

/**
 * Extract candidate image file-path tokens from free text. Matches path-like
 * tokens (absolute `/…`, home `~/…`, or relative `./…`/`../…`) that end in a
 * known image extension. Bare filenames without a path separator are
 * deliberately not matched to avoid false positives on ordinary words. URLs
 * (`http://…`) can match the pattern but are filtered out later by the
 * `existsSync` check in `resolveImageFile`.
 */
export function findImagePathTokens(text: string): string[] {
  const matches: Array<{ token: string; index: number }> = [];
  for (const pattern of [PATH_TOKEN_RE, WINDOWS_PATH_TOKEN_RE]) {
    pattern.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text)) !== null) {
      matches.push({ token: match[0], index: match.index });
    }
  }
  matches.sort((a, b) => a.index - b.index);
  return [...new Set(matches.map((match) => match.token))];
}

/** Resolve a token against cwd and return the absolute path if it's a real
 *  image file, else undefined. Unescapes \ (escaped spaces from terminal
 *  drag-and-drop) before resolving. */
function resolveImageFile(token: string, cwd: string): string | undefined {
  // Unescape \ → space (terminal drag-paste escaping)
  const unescaped = token.replace(/\\ /g, " ");
  const expanded = unescaped.startsWith("~/") ? resolvePath(cwd, unescaped) : unescaped;
  const abs = isAbsolute(expanded) ? expanded : resolvePath(cwd, expanded);
  if (!existsSync(abs)) return undefined;
  try {
    if (!statSync(abs).isFile()) return undefined;
  } catch {
    return undefined;
  }
  if (!IMAGE_EXT_RE.test(abs)) return undefined;
  return abs;
}

/** Cheap FNV-1a hash over the base64 string, for dedup against existing
 *  attachments (e.g. images pi-paster already attached). */
function hashData(data: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < data.length; i++) {
    h ^= data.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16);
}

/** A loaded image ready for attachment or delegation. */
interface LoadedImage {
  token: string;
  abs: string;
  data: string;
  mimeType: string;
  hash: string;
}

/** Load + dedup all resolvable tokens. Returns the loaded images + the
 *  set of hashes already seen (for dedup). */
async function loadAndDedup(
  tokens: string[],
  existingImages: ImageContent[],
  cwd: string,
): Promise<{ loaded: LoadedImage[]; existingHashes: Set<string> }> {
  const existingHashes = new Set(existingImages.map((img) => hashData(img.data)));
  const loaded: LoadedImage[] = [];
  const seenPaths = new Set<string>();

  for (const token of tokens) {
    const abs = resolveImageFile(token, cwd);
    if (!abs || seenPaths.has(abs)) continue;
    seenPaths.add(abs);
    const result = await loadImage(abs, {
      compress: true,
      maxDimension: 1568,
      jpegQuality: 85,
      cwd,
    });
    if (!result.ok) continue; // skip unreadable images
    const hash = hashData(result.image.data);
    if (existingHashes.has(hash)) continue; // already attached (e.g. by pi-paster)
    existingHashes.add(hash);
    loaded.push({ token, abs, data: result.image.data, mimeType: result.image.mimeType, hash });
  }

  return { loaded, existingHashes };
}

/** Build the resolved map for renderMarkers: token → index in the final
 *  images array. For multimodal, index = offset + position among loaded.
 *  For dedup'd images (hash matches existing), the marker points to the
 *  existing index. For text-only (no attachment), markers are sequential
 *  starting from the offset. */
function buildResolvedMap(
  tokens: string[],
  loaded: LoadedImage[],
  existingImages: ImageContent[],
  isMultimodalModel: boolean,
): Map<string, { index: number }> {
  const resolved = new Map<string, { index: number }>();
  const offset = existingImages.length;

  // For dedup: build a hash → existing-index lookup.
  const existingHashToIndex = new Map<string, number>();
  existingImages.forEach((img, i) => {
    existingHashToIndex.set(hashData(img.data), i);
  });

  let newImageIndex = 0;
  for (const token of tokens) {
    // Find if this token was loaded.
    const found = loaded.find((l) => l.token === token);
    if (!found) continue; // unresolvable → not in resolved map → left as-is

    // Check if it was dedup'd (hash matches an existing image).
    const existingIdx = existingHashToIndex.get(found.hash);
    if (existingIdx !== undefined) {
      resolved.set(token, { index: existingIdx });
    } else {
      // New image.
      const idx = isMultimodalModel ? offset + newImageIndex : newImageIndex;
      resolved.set(token, { index: idx });
      newImageIndex++;
    }
  }

  return resolved;
}

/** Auto-delegate a single image with timeout protection. Returns the
 *  description text or undefined on failure/timeout (caller falls back to hint).
 *  v0.4.0: takes the shared batch `signal` (owned by the caller) instead of
 *  creating its own AbortController — the batch owns one timeout for all
 *  images, bounding the wall-clock of the pre-send input hook. */
async function autoDelegateOne(
  ctx: ExtensionContext,
  config: NonNullable<ReturnType<typeof getSharedConfig>>,
  image: LoadedImage,
  cache: NonNullable<ReturnType<typeof getSharedCache>>,
  signal: AbortSignal,
): Promise<{ text: string; cached: boolean } | undefined> {
  try {
    const params: DelegateParams = {
      image_path: image.abs,
      prompt: config.autoDelegatePrompt,
      compress: true,
      reasoning: "off" as ReasoningLevel,
    };
    const result = await delegateToVisionModel(ctx, config, params, signal, cache);
    if (result.ok) {
      return { text: result.text, cached: result.details.cached };
    }
    return undefined; // failure → caller falls back to hint
  } catch {
    return undefined; // timeout or error → hint fallback
  }
}

/** Debounce timer for compose-time preview. */
let composeDebounceTimer: ReturnType<typeof setTimeout> | undefined;
/** Unsubscribe function for the onTerminalInput handler. */
let terminalInputUnsub: (() => void) | undefined;
/** The last set of tokens we previewed (to avoid re-rendering the same). */
let lastPreviewTokens: string = "";
const composeImageReferences = new ComposeImageReferences();
let previousEditorFactory: EditorFactory | undefined;
let installedEditorFactory: EditorFactory | undefined;

/**
 * Compose-time auto-preview: while the user is typing, detect image paths in
 * the editor text and render a preview above the editor (WhatsApp style).
 * Debounced ~300ms after the last keystroke. Clears on submit or path removal.
 */
async function updateComposePreview(ctx: ExtensionContext): Promise<void> {
  const config = getSharedConfig();
  if (!config || !config.composePreview) {
    ctx.ui.setWidget("vision-compose-preview", undefined);
    return;
  }
  if (ctx.mode !== "tui") return; // widget is TUI-only

  const editorText = composeImageReferences.expand(ctx.ui.getEditorText());
  // Don't show compose preview for slash commands — they have their own UI
  if (editorText.trim().startsWith("/")) {
    if (lastPreviewTokens) {
      ctx.ui.setWidget("vision-compose-preview", undefined);
      lastPreviewTokens = "";
    }
    return;
  }
  const tokens = findImagePathTokens(editorText);
  const tokensKey = tokens.join("\0");

  if (tokens.length === 0) {
    if (lastPreviewTokens) {
      ctx.ui.setWidget("vision-compose-preview", undefined);
      lastPreviewTokens = "";
    }
    return;
  }

  // Skip if same tokens as last time (no re-render needed)
  if (tokensKey === lastPreviewTokens) return;
  lastPreviewTokens = tokensKey;

  // Resolve + load each token (compress: false — show original quality)
  const previewImages: ReturnType<typeof makePreviewImage>[] = [];
  for (const token of tokens) {
    const unescaped = token.replace(/\\ /g, " ");
    const abs = isAbsolute(unescaped) ? unescaped : resolvePath(ctx.cwd, unescaped);
    if (!existsSync(abs)) continue;
    const result = await loadImage(abs, {
      compress: false,
      maxDimension: 1568,
      jpegQuality: 85,
      cwd: ctx.cwd,
    });
    if (!result.ok) continue;
    previewImages.push(makePreviewImage(result.image.data, result.image.mimeType, abs));
  }

  if (previewImages.length === 0) {
    ctx.ui.setWidget("vision-compose-preview", undefined);
    return;
  }

  // Set the widget above the editor
  ctx.ui.setWidget(
    "vision-compose-preview",
    (tui, theme) => createComposePreviewComponent(
      previewImages,
      (c: string, t: string) => theme.fg(c as any, t),
      config.previewMaxWidthCells,
    ),
    { placement: "aboveEditor" },
  );
}

export default function pasteExtension(_pi: ExtensionAPI): void {
  // ── Compose-time auto-preview (gap #7) ───────────────────────────────────
  _pi.on("session_start", (_event, ctx) => {
    // Register terminal input listener for compose-time preview
    if (terminalInputUnsub) terminalInputUnsub();
    composeImageReferences.reset();
    if (ctx.hasUI && ctx.mode === "tui") {
      const currentFactory = ctx.ui.getEditorComponent();
      const baseFactory = currentFactory === installedEditorFactory
        ? previousEditorFactory
        : currentFactory;
      const wrapperFactory: EditorFactory = (tui, theme, keybindings) => {
        const editor = baseFactory?.(tui, theme, keybindings)
          ?? new CustomEditor(tui, theme, keybindings);
        return addComposeImageMarkersToEditor(
          editor,
          composeImageReferences,
          (marker) => ctx.ui.theme.fg("accent", marker),
        );
      };
      previousEditorFactory = baseFactory;
      installedEditorFactory = wrapperFactory;
      ctx.ui.setEditorComponent(wrapperFactory);

      terminalInputUnsub = ctx.ui.onTerminalInput((_data: string) => {
        // Debounce: clear existing timer, set new one
        if (composeDebounceTimer) clearTimeout(composeDebounceTimer);
        composeDebounceTimer = setTimeout(() => {
          updateComposePreview(ctx).catch(() => {});
        }, 300);
        return remapMacOptionVPaste(_data);
      });
    }
  });

  _pi.on("session_shutdown", (_event, ctx) => {
    if (composeDebounceTimer) {
      clearTimeout(composeDebounceTimer);
      composeDebounceTimer = undefined;
    }
    if (terminalInputUnsub) {
      terminalInputUnsub();
      terminalInputUnsub = undefined;
    }
    if (
      installedEditorFactory
      && ctx.mode === "tui"
      && typeof ctx.ui.getEditorComponent === "function"
      && ctx.ui.getEditorComponent() === installedEditorFactory
    ) {
      ctx.ui.setEditorComponent(previousEditorFactory);
    }
    previousEditorFactory = undefined;
    installedEditorFactory = undefined;
    lastPreviewTokens = "";
    composeImageReferences.reset();
  });

  _pi.on("input", async (event, ctx) => {
    // Clear compose preview on submit (only if setWidget is available — TUI mode)
    if (event.source !== "extension" && typeof ctx.ui.setWidget === "function") {
      ctx.ui.setWidget("vision-compose-preview", undefined);
      lastPreviewTokens = "";
    }
    // Don't re-process messages we (or another extension) injected.
    if (event.source === "extension") return { action: "continue" as const };

    const config = getSharedConfig();
    if (!config) return { action: "continue" as const }; // before session_start

    const expandedText = composeImageReferences.expand(event.text);
    const tokens = findImagePathTokens(expandedText);
    if (tokens.length === 0) {
      composeImageReferences.reset();
      return { action: "continue" as const };
    }

    const multimodal = isMultimodal(ctx.model);
    const { loaded, existingHashes } = await loadAndDedup(
      tokens,
      event.images ?? [],
      ctx.cwd,
    );
    const firstUnresolvedIndex = (multimodal ? (event.images?.length ?? 0) : 0) + loaded.length + 1;

    if (loaded.length === 0) {
      const text = composeImageReferences.restorePaths(
        expandedText,
        (index) => styleMarker(index, config.markerStyle),
        firstUnresolvedIndex,
      );
      composeImageReferences.reset();
      return text === event.text
        ? { action: "continue" as const }
        : { action: "transform" as const, text };
    }

    // Build the resolved map (token → index for marker numbering).
    const resolved = buildResolvedMap(tokens, loaded, event.images ?? [], multimodal);

    // Render markers in the text.
    let text = renderMarkers(expandedText, tokens, resolved, config.markerStyle);
    text = composeImageReferences.restorePaths(
      text,
      (index) => styleMarker(index, config.markerStyle),
      firstUnresolvedIndex,
    );
    composeImageReferences.reset();

    if (multimodal) {
      // ── MULTIMODAL: attach images + markers (B-lite delivery + gap #6) ──
      const newImages: ImageContent[] = loaded.map((l) => ({
        type: "image" as const,
        data: l.data,
        mimeType: l.mimeType,
      }));
      return {
        action: "transform" as const,
        text,
        images: [...(event.images ?? []), ...newImages],
      };
    }

    // ── TEXT-ONLY: no attachment (text-only models can't process images) ──
    const mode = config.textOnlyPasteMode;

    if (mode === "off") {
      // Markers only — no attachment, no hint, no delegation.
      return { action: "transform" as const, text };
    }

    if (mode === "hint") {
      // Markers + hint line nudging the model to call describe_image.
      text = `${text}\n${buildHintLine(loaded.map((l, i) => ({ token: l.token, index: resolved.get(l.token)?.index ?? i })))}`;
      return { action: "transform" as const, text };
    }

    // mode === "auto": auto-delegate each image in PARALLEL (v0.4.0 SPEC-4 §3.2)
    // with bounded concurrency + ONE batch-level AbortController (the budget
    // bounds the pre-send wall-clock, not per-image). Reuses the v0.2.x delegate
    // pipeline (cache/retry/fallback) per image. Falls back to hint on timeout/
    // failure (all-fail → hint; per-image fail → that image gets no description).
    const cache = getSharedCache();
    const visionModel = config.provider && config.model ? `${config.provider}/${config.model}` : "(unconfigured)";
    const hintImages = loaded.map((l, i) => ({ token: l.token, index: resolved.get(l.token)?.index ?? i }));

    // ── Local-only short-circuit (SPEC-5 §3.2) ────────────────────────────
    // If local-only mode is on, every delegation would be refused (cache miss)
    // or cache-only. Skip the batch entirely — don't burn autoDelegateTimeoutMs
    // waiting for refused calls to abort. Fall straight to hint so the model
    // can still call describe_image for cache hits (which local-only allows).
    if (config.localOnly) {
      text = `${text}\n${buildHintLine(hintImages)}`;
      return { action: "transform" as const, text };
    }

    if (!cache || !config.provider || !config.model) {
      // Can't delegate (no cache or unconfigured) → fall back to hint.
      text = `${text}\n${buildHintLine(hintImages)}`;
      return { action: "transform" as const, text };
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), config.autoDelegateTimeoutMs);
    let results: Array<{ text: string; cached: boolean } | undefined>;
    try {
      results = await mapWithConcurrency(
        loaded,
        config.batchConcurrency,
        (image) => autoDelegateOne(ctx, config, image, cache, controller.signal),
      );
    } finally {
      clearTimeout(timer);
    }

    const descriptions: Array<{ token: string; index: number; text: string; cached: boolean }> = [];
    let ok = 0;
    for (let i = 0; i < loaded.length; i++) {
      const r = results[i];
      if (r) {
        descriptions.push({ token: loaded[i]!.token, index: resolved.get(loaded[i]!.token)?.index ?? i, text: r.text, cached: r.cached });
        ok++;
      }
      // undefined → that image gets no description (timeout/failure mid-batch)
    }

    if (ok === 0) {
      // All failed/timed out → hint fallback (with paths, §3.4).
      text = `${text}\n${buildHintLine(hintImages)}`;
      return { action: "transform" as const, text };
    }

    // Append the descriptions block.
    text = `${text}${buildDescriptionsBlock(descriptions, visionModel)}`;
    return { action: "transform" as const, text };
  });
}