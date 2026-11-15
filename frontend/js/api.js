// Talks to the SPHEREx Blink backend.
// Developed with AI assistance (Claude, Anthropic). See README "Use of AI".

function url(path, params) {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== null && value !== undefined && value !== "") qs.set(key, value);
  }
  const text = qs.toString();
  return text ? `${path}?${text}` : path;
}

async function failure(resp) {
  let detail = `${resp.status} ${resp.statusText}`;
  try {
    const body = await resp.json();
    if (body && body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch { /* not JSON */ }
  const err = new Error(detail);
  err.status = resp.status;
  return err;
}

export async function getJSON(path, params, signal) {
  const resp = await fetch(url(path, params), { signal });
  if (!resp.ok) throw await failure(resp);
  return resp.json();
}

/**
 * Fetch aligned frames. The payload is:
 *   'SPXB' | uint32 header length | JSON header | float32 frames | uint8 flag planes
 */
export async function getFrames(params, signal, path = "api/frames") {
  const resp = await fetch(url(path, params), { signal });
  if (!resp.ok) throw await failure(resp);
  const buffer = await resp.arrayBuffer();
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== "SPXB") throw new Error("Unexpected response from the server");
  const headerLength = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const plane = header.n * header.n;
  const count = header.nframes * plane;                      // image planes (3 per frame for colour)
  const flagCount = (header.nflags ?? header.nframes) * plane;
  const offset = 8 + headerLength;
  // Copy out of the response buffer so each frame is an independent typed array.
  const data = new Float32Array(buffer.slice(offset, offset + count * 4));
  const flags = new Uint8Array(buffer.slice(offset + count * 4, offset + count * 4 + flagCount));
  return { header, data, flags };
}
