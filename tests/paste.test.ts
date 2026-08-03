import { test } from "node:test";
import assert from "node:assert/strict";
import {
  addComposeImageMarkersToEditor,
  ComposeImageReferences,
  findImagePathTokens,
  remapMacOptionVPaste,
} from "../extensions/paste.ts";

test("addComposeImageMarkersToEditor: colors markers only in rendered output", () => {
  let text = "compare [Image-#1] and [Image-#2]";
  const editor = {
    getText: () => text,
    setText: (next: string) => { text = next; },
    insertTextAtCursor: (inserted: string) => { text += inserted; },
    handleInput: () => {},
    render: () => [text],
    invalidate: () => {},
  };
  const references = new ComposeImageReferences();
  const wrapped = addComposeImageMarkersToEditor(
    editor,
    references,
    (marker) => `<accent>${marker}</accent>`,
  );

  assert.deepEqual(
    wrapped.render(80),
    ["compare <accent>[Image-#1]</accent> and <accent>[Image-#2]</accent>"],
  );
  assert.equal(wrapped.getText(), "compare [Image-#1] and [Image-#2]");
});

test("ComposeImageReferences: replaces clipboard paths with visible image markers", () => {
  const refs = new ComposeImageReferences();
  const first = "/tmp/pi-clipboard-11111111-1111-4111-8111-111111111111.png";
  const second = "/tmp/pi-clipboard-22222222-2222-4222-8222-222222222222.jpg";

  assert.equal(refs.collapse(`compare ${first}`), "compare [Image-#1]");
  assert.equal(refs.collapse(`compare [Image-#1] and ${second}`), "compare [Image-#1] and [Image-#2]");
  assert.equal(
    refs.expand("compare [Image-#1] and [Image-#2]"),
    `compare ${first} and ${second}`,
  );
});

test("ComposeImageReferences: ignores regular paths and avoids handwritten marker numbers", () => {
  const refs = new ComposeImageReferences();
  const clipboard = "/tmp/pi-clipboard-33333333-3333-4333-8333-333333333333.webp";

  assert.equal(refs.collapse("see /tmp/regular.png"), "see /tmp/regular.png");
  assert.equal(refs.collapse(`literal [Image-#1] ${clipboard}`), "literal [Image-#1] [Image-#2]");
  refs.reset();
  assert.equal(refs.collapse(clipboard), "[Image-#1]");
});

test("ComposeImageReferences: reuses numbering after the marker is deleted", () => {
  const refs = new ComposeImageReferences();
  const first = "/tmp/pi-clipboard-44444444-4444-4444-8444-444444444444.png";
  const replacement = "/tmp/pi-clipboard-55555555-5555-4555-8555-555555555555.png";

  assert.equal(refs.collapse(first), "[Image-#1]");
  assert.equal(refs.collapse(""), "");
  assert.equal(refs.collapse(replacement), "[Image-#1]");
});

test("ComposeImageReferences: collapses Windows clipboard paths passed by the editor", () => {
  const refs = new ComposeImageReferences();
  const clipboard = String.raw`C:\Users\pi\AppData\Local\Temp\pi-clipboard-66666666-6666-4666-8666-666666666666.png`;

  assert.equal(refs.collapseInsertion(clipboard, ""), "[Image-#1]");
  assert.equal(refs.expand("[Image-#1]"), clipboard);
});

test("ComposeImageReferences: restores unresolved expanded paths to their markers", () => {
  const refs = new ComposeImageReferences();
  const clipboard = "/tmp/pi-clipboard-88888888-8888-4888-8888-888888888888.png";
  refs.collapseInsertion(clipboard, "");

  assert.equal(
    refs.restorePaths(`failed ${clipboard}`, (index) => `**[Image-#${index}]**`),
    "failed **[Image-#1]**",
  );
});

test("remapMacOptionVPaste: macOS Option+V forwards to pi's image-paste input", () => {
  assert.deepEqual(remapMacOptionVPaste("\x1bv", "darwin"), { data: "\x16" });
  assert.deepEqual(remapMacOptionVPaste("\x1b[118;3u", "darwin"), { data: "\x16" });
});

test("remapMacOptionVPaste: leaves other keys and platforms unchanged", () => {
  assert.equal(remapMacOptionVPaste("v", "darwin"), undefined);
  assert.equal(remapMacOptionVPaste("\x1bV", "darwin"), undefined);
  assert.equal(remapMacOptionVPaste("\x1b[118;3:3u", "darwin"), undefined);
  assert.equal(remapMacOptionVPaste("\x1bv", "linux"), undefined);
  assert.equal(remapMacOptionVPaste("\x1bv", "win32"), undefined);
});

test("findImagePathTokens: absolute path", () => {
  assert.deepEqual(findImagePathTokens("analyze /tmp/screenshot.png"), ["/tmp/screenshot.png"]);
});

test("findImagePathTokens: home path", () => {
  assert.deepEqual(findImagePathTokens("see ~/Pictures/diagram.jpg"), ["~/Pictures/diagram.jpg"]);
});

test("findImagePathTokens: relative ./ and ../ paths", () => {
  assert.deepEqual(findImagePathTokens("./a.png and ../b/c.gif"), ["./a.png", "../b/c.gif"]);
});

test("findImagePathTokens: multiple paths in one message", () => {
  const out = findImagePathTokens("compare /x.png and /y.jpeg please");
  assert.deepEqual(out, ["/x.png", "/y.jpeg"]);
});

test("findImagePathTokens: dedups repeated paths", () => {
  const out = findImagePathTokens("/x.png then /x.png again");
  assert.deepEqual(out, ["/x.png"]);
});

test("findImagePathTokens: bare filename without separator is NOT matched", () => {
  assert.deepEqual(findImagePathTokens("look at photo.png for me"), []);
  assert.deepEqual(findImagePathTokens("photo.png"), []);
});

test("findImagePathTokens: non-image extension is NOT matched", () => {
  assert.deepEqual(findImagePathTokens("/tmp/notes.txt"), []);
  assert.deepEqual(findImagePathTokens("/tmp/data.json"), []);
});

test("findImagePathTokens: all supported extensions", () => {
  const out = findImagePathTokens("/a.png /b.jpg /c.jpeg /d.gif /e.webp /f.bmp");
  assert.equal(out.length, 6);
});

test("findImagePathTokens: empty / no paths", () => {
  assert.deepEqual(findImagePathTokens(""), []);
  assert.deepEqual(findImagePathTokens("just a normal message with no images"), []);
});

test("findImagePathTokens: path adjacent to punctuation", () => {
  const out = findImagePathTokens("see (/tmp/x.png) and /tmp/y.png.");
  assert.ok(out.includes("/tmp/x.png"));
  assert.ok(out.some((p) => p.startsWith("/tmp/y.png")));
});

test("findImagePathTokens: URL with image ext is matched by regex but filtered later", () => {
  // The regex matches the path-like portion; resolveImageFile's existsSync
  // check filters non-local paths. Here we only assert the regex surfaces it.
  const out = findImagePathTokens("https://example.com/cat.png");
  assert.ok(out.length >= 1, "regex surfaces URL path-like token; existsSync filters it downstream");
});
test("findImagePathTokens: escaped-space path (terminal drag-paste on macOS)", () => {
  const out = findImagePathTokens("/var/folders/NSIRD_screencaptureui/Screenshot\\ 2026-07-10\\ at\\ 10.30.43.png");
  assert.equal(out.length, 1);
  assert.ok(out[0]!.includes("Screenshot\\ 2026-07-10\\ at\\ 10.30.43.png"));
});

test("findImagePathTokens: escaped-space path followed by more text", () => {
  const out = findImagePathTokens("analyze /tmp/My\\ Screenshot.png for bugs");
  assert.equal(out.length, 1);
  assert.equal(out[0], "/tmp/My\\ Screenshot.png");
});

test("findImagePathTokens: escaped + regular paths mixed", () => {
  const out = findImagePathTokens("compare /tmp/a.png and /tmp/My\\ B.jpeg");
  assert.equal(out.length, 2);
  assert.equal(out[0], "/tmp/a.png");
  assert.equal(out[1], "/tmp/My\\ B.jpeg");
});

test("findImagePathTokens: Windows clipboard path with spaces", () => {
  const path = String.raw`C:\Users\Pi User\AppData\Local\Temp\pi-clipboard-77777777-7777-4777-8777-777777777777.png`;
  assert.deepEqual(findImagePathTokens(`analyze ${path}`), [path]);
});
