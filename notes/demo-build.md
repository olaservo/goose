# Building the packaged demo app (`C:\goose-demo`)

Written 2026-09-17 from the build of `skills-over-mcp` at `e69badcb36`. Run on Windows 11 from this clone (`_bellafoo/goose`), Git Bash.

## Steps

1. Build the release CLI binary into a short target dir. About 14 minutes cold.

   ```bash
   CARGO_TARGET_DIR=C:/gt cargo build --release -p goose-cli --bin goose
   ```

2. Copy it into the desktop app's bin folder.

   ```bash
   cp /c/gt/release/goose.exe ui/desktop/src/bin/goose.exe
   ```

3. First time in a clone only: install UI dependencies from `ui/`.

   ```bash
   cd ui && pnpm install --frozen-lockfile
   ```

4. Package from `ui/desktop`. The first command downloads the pinned `uv.exe`/`uvx.exe`.

   ```bash
   cd ui/desktop
   export ELECTRON_PLATFORM=win32
   node scripts/prepare-platform-binaries.js
   pnpm run package --platform=win32 --arch=x64
   ```

   Output is `ui/desktop/out/Goose-win32-x64/` (gitignored), with the CLI at `resources/bin/goose.exe`.

5. Swap the demo folder. Close Goose first. Keep the previous build as a dated backup.

   ```bash
   mv /c/goose-demo /c/goose-demo.bak-<date>
   cp -r ui/desktop/out/Goose-win32-x64 /c/goose-demo
   /c/goose-demo/resources/bin/goose.exe --version
   ```

## Notes

- `pnpm run package` produces the unpacked app folder, which is what `C:\goose-demo` is. `just make-ui-windows` is the upstream recipe; it builds for the `x86_64-pc-windows-msvc` target under `./target` and runs `pnpm run make`, which also builds installers. It was not used here.
- `C:\gt` was already in use as the target dir on 2026-09-03. The reason was not recorded. Debug test builds also go there (`C:\gt\debug`).
- The app reads `%APPDATA%\Block\goose\config\config.yaml`, shared with any other goose install on the machine.
- To swap only the CLI binary without repackaging, copy `C:\gt\release\goose.exe` over `C:\goose-demo\resources\bin\goose.exe`. UI changes need the full package step.
- Backup of the 2026-09-03 build: `C:\goose-demo.bak-2026-09-03` (684 MB). Delete it once the new build is confirmed.
- Rebuilt 2026-09-24 from `e69badcb36` plus the uncommitted seal and attribution changes in the working tree. Incremental release build took 19m 44s. Backup of the 2026-09-17 build: `C:\goose-demo.bak-2026-09-17`.
- Rebuilt 2026-09-28 from `e69badcb36` plus the interceptor-client changes (grading moved to the Interceptor Server), committed as `cec081d94f` on branch `attribution-interceptors`, cut from `skills-over-mcp` at the same commit and pushed to `olaservo/goose`. `skills-over-mcp` itself is unchanged. Release builds took 13m 19s and, after the `interceptors/list` probe fix, 11m 34s. Only the CLI was swapped, into `C:\goose-demo\resources\bin\goose.exe`, with the previous binary beside it as `goose.exe.bak-2026-09-28`; the UI was unchanged since 9/24. The same binary is staged at `ui/desktop/src/bin/goose.exe`.

## Attribution demo

- Start a server that serves skills, for example `bun main.ts` in `my-projects/cthulhu-keeper-mcp` (`:3002`, extension `cthulhu_keeper`, `skills_enabled: true`).
- Since 2026-09-24 the `cthulhu_keeper` extension in `config.yaml` points at the private Space `https://olaservo-cthulhu-keeper-mcp.hf.space/mcp` with header `Authorization: Bearer ${HF_TOKEN}` and `env_keys: [HF_TOKEN]`. goose resolves `HF_TOKEN` from the environment first, then its keyring. The Hugging Face provider signs in over OAuth, so it does not put `HF_TOKEN` in the keyring; either launch with `HF_TOKEN=$(hf auth token) /c/goose-demo/Goose.exe` or store the secret once from the app's extension settings. To use the LAN server instead, set `uri` back to `http://localhost:3002/mcp` and clear the header. Previous config is `config.yaml.bak-20260924`.
- Launch `C:\goose-demo\Goose.exe`, start a chat, then open Skills. MCP skills list only while a session with that extension is open.
- Each MCP skill shows `MCP · <server>`, a badge (`attributed`, `partial`, `uncredited`) and a credit summary.
- Set `GOOSE_SKILLS_ATTRIBUTION_REQUIRE=true` to make `load_skill` withhold an uncredited skill.
- Since 2026-09-28 grading is not in goose. It runs on an Interceptor Server, `my-projects/attribution-interceptors/server.py`, configured in `config.yaml` as the stdio extension `attribution_interceptors` (`uv run --project <repo> server.py`). goose finds it by the `io.modelcontextprotocol/interceptors` capability and sends `interceptor/invoke` for `seal` and `attribution` on every `load_skill`. Without that extension skills load ungraded (`unchecked`). Previous config is `config.yaml.bak-2026-09-28`.
- The `spaceship_server` extension now launches `proxy.py --mode audit` from the same repo, which spawns `spaceship-server/dist/index.js` and invokes the Interceptor Server on `skills/list`, `skills/get` and `resources/read` responses. It writes `attribution-interceptors/gateway.jsonl`. Change `audit` to `active` in `config.yaml` to make the gateway block a rejected response, then restart goose.
- Set `GOOSE_SKILLS_ACTIVATION_LOG=<path>` to get one JSON line per `load_skill` with the interceptor verdicts.

## Screenshots without clicking through the app

Run on 2026-09-17 against `C:\goose-demo`. Electron on Windows still opens a window; nothing needs clicking.

1. Start the skills server (`bun main.ts` in `my-projects/cthulhu-keeper-mcp`).
2. Launch the app with its debugging port open: `ENABLE_PLAYWRIGHT=true PLAYWRIGHT_DEBUG_PORT=9333 /c/goose-demo/Goose.exe`. `src/main.ts` adds `--remote-debugging-port` when `ENABLE_PLAYWRIGHT` is set, and this works in the packaged build.
3. `node notes/scripts/skills-view-session.cjs <out-dir>` connects over CDP, sends one chat message to open a session (one request to the configured model), opens `#/skills` and screenshots it.
4. `node notes/scripts/skills-view-rows.cjs <out-dir>` screenshots each MCP skill row and prints its tag, badge and credit line.

The scripts load Playwright from `ui/node_modules/@playwright/test` by absolute path. Output from the first run is in `notes/screenshots-2026-09-17/`.

Observed on that run: four Cthulhu content skills graded `attributed`, `cthulhu-keeper-guide` graded `partial`, and the three `skills-over-mcp-demo` skills graded `partial` (author only). No served skill graded `uncredited`, so the red badge and the withhold path have unit tests only.
