/**
 * Framer Code Component: Purchasing Automation
 * Paste this file into Framer as a Code component.
 * In Framer, add a string property: apiBaseUrl (e.g. https://your-api.com). Leave empty if same origin or proxy.
 */
import React, { useCallback, useRef, useState } from "react"
import { addPropertyControls, ControlType } from "framer"

type DownloadItem = { filename: string; content_base64?: string | null }

function downloadFile(
    item: DownloadItem,
    onError?: (message: string) => void
): void {
    if (!item.content_base64) {
        onError?.(
            `Download failed: ${item.filename} has no content (content_base64 missing). Run the pipeline again.`
        )
        return
    }
    try {
        const bin = atob(item.content_base64)
        const bytes = new Uint8Array(bin.length)
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
        const blob = new Blob([bytes], {
            type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        })
        const a = document.createElement("a")
        const url = URL.createObjectURL(blob)
        a.href = url
        a.download = item.filename
        a.style.display = "none"
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        setTimeout(() => URL.revokeObjectURL(url), 500)
    } catch (e) {
        onError?.(`Download failed: ${item.filename} — ${String(e)}`)
    }
}

// --- File system types for folder drop ---
interface FileSystemFileEntry {
    isFile: true
    isDirectory: false
    name: string
    file(success: (f: File) => void, error?: (e: Error) => void): void
}
interface FileSystemDirectoryEntry {
    isFile: false
    isDirectory: true
    name: string
    createReader(): FileSystemDirectoryReader
}
interface FileSystemDirectoryReader {
    readEntries(
        success: (
            entries: (FileSystemFileEntry | FileSystemDirectoryEntry)[]
        ) => void,
        error?: (e: Error) => void
    ): void
}
/**
 * Collects PDF files from a Drag & Drop event, including traversing folders.
 */
async function collectFilesFromDrop(
    items: DataTransferItemList,
    acceptExt: string[] = [".pdf"]
): Promise<File[]> {
    const files: File[] = []

    const isAccepted = (name: string) => {
        if (!acceptExt.length) return true
        const ext = "." + (name.split(".").pop() || "").toLowerCase()
        return acceptExt.some((e) => e.toLowerCase() === ext)
    }

    // 1. Get entries synchronously to avoid data loss after the event loop
    const topLevelEntries: any[] = []
    for (let i = 0; i < items.length; i++) {
        const item = items[i]
        if (item.kind !== "file") continue
        const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null
        if (entry) {
            topLevelEntries.push(entry)
        } else {
            // Fallback for simple files if entry is not available
            const file = item.getAsFile()
            if (file && isAccepted(file.name)) files.push(file)
        }
    }

    // 2. Recursive traversal function
    async function traverse(entry: any) {
        if (entry.isFile) {
            return new Promise<void>((resolve) => {
                entry.file(
                    (file: File) => {
                        if (isAccepted(file.name)) files.push(file)
                        resolve()
                    },
                    () => resolve()
                )
            })
        } else if (entry.isDirectory) {
            const reader = entry.createReader()
            let allSubEntries: any[] = []

            // readEntries only returns up to 100 items; must loop until empty
            const readAll = (): Promise<void> => {
                return new Promise((resolve, reject) => {
                    reader.readEntries(
                        (results: any[]) => {
                            if (results.length === 0) {
                                resolve()
                            } else {
                                allSubEntries = allSubEntries.concat(results)
                                readAll().then(resolve).catch(reject)
                            }
                        },
                        (err: Error) => reject(err)
                    )
                })
            }

            try {
                await readAll()
                await Promise.all(allSubEntries.map((e) => traverse(e)))
            } catch (e) {
                console.error("Folder traversal error:", e)
            }
        }
    }

    // 3. Process all top-level entries
    await Promise.all(topLevelEntries.map((e) => traverse(e)))
    return files
}

// --- Ingest ---
const INGEST_TIMEOUT_MS = 90000 // 90s so request does not hang forever (e.g. CORS or wrong apiBaseUrl in Framer)
type IngestType =
    | "supplier-history"
    | "item-history"
    | "analysis-examples"
    | "request-examples"
    | "email-examples"

function FolderDropZone({
    apiBase,
    apiToken,
    ingestType,
    label,
}: {
    apiBase: string
    apiToken: string
    ingestType: IngestType
    label: string
}) {
    const [dragging, setDragging] = useState(false)
    const [uploading, setUploading] = useState(false)
    const [result, setResult] = useState<{
        ok: boolean
        processed?: number
        results?: unknown[]
        error?: string
    } | null>(null)
    const [accumulatedFiles, setAccumulatedFiles] = useState<Map<string, File>>(
        () => new Map()
    )
    const accumulatedRef = useRef<Map<string, File>>(new Map())

    const uploadFiles = useCallback(
        async (files: File[]) => {
            if (files.length === 0) {
                setResult({ ok: false, error: "No PDF files found." })
                return
            }
            setUploading(true)
            setResult(null)
            const form = new FormData()
            files.forEach((f) => form.append("files", f))
            const controller = new AbortController()
            const timeoutId = setTimeout(
                () => controller.abort(),
                INGEST_TIMEOUT_MS
            )
            try {
                const url = apiBase
                    ? `${apiBase.replace(/\/$/, "")}/api/ingest/${ingestType}`
                    : `/api/ingest/${ingestType}`
                const res = await fetch(url, {
                    method: "POST",
                    body: form,
                    headers: {
                        "X-API-Key": apiToken,
                    },
                    signal: controller.signal,
                })
                clearTimeout(timeoutId)
                const text = await res.text()
                const data = text
                    ? (() => {
                        try {
                            return JSON.parse(text)
                        } catch {
                            return {}
                        }
                    })()
                    : {}
                if (!res.ok) {
                    const detail =
                        (data as { detail?: string }).detail || res.statusText
                    const msg =
                        res.status === 0
                            ? "Request blocked (check apiBaseUrl and CORS)."
                            : `Upload failed (${res.status}): ${detail}`
                    setResult({ ok: false, error: msg })
                    return
                }
                setResult({
                    ok: true,
                    processed: (data as { processed?: number }).processed,
                    results: (data as { results?: unknown[] }).results,
                })
            } catch (e) {
                clearTimeout(timeoutId)
                const err = e as Error & { name?: string }
                const msg =
                    err.name === "AbortError"
                        ? "Upload timed out. Check apiBaseUrl and CORS for your Framer site."
                        : String(e)
                setResult({ ok: false, error: msg })
            } finally {
                setUploading(false)
            }
        },
        [apiBase, ingestType]
    )

    const onDrop = useCallback(
        async (e: React.DragEvent) => {
            e.preventDefault()
            setDragging(false)
            const items = e.dataTransfer?.items
            if (!items?.length) return
            const newFiles = await collectFilesFromDrop(items, [".pdf"])
            if (newFiles.length === 0) {
                setResult({ ok: false, error: "No PDF files found." })
                return
            }
            const merged = new Map(accumulatedRef.current)
            newFiles.forEach((f) => merged.set(f.name, f))
            accumulatedRef.current = merged
            setAccumulatedFiles(merged)
            await uploadFiles(Array.from(merged.values()))
        },
        [uploadFiles]
    )

    const onDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.dataTransfer.dropEffect = "copy"
        setDragging(true)
    }, [])

    const onDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setDragging(false)
    }, [])

    return (
        <div
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            style={{
                border: `2px dashed ${dragging ? "#07c" : "#ccc"}`,
                borderRadius: 8,
                padding: 24,
                textAlign: "center",
                background: dragging ? "#f0f8ff" : "#fafafa",
                cursor: uploading ? "wait" : "default",
                opacity: uploading ? 0.8 : 1,
            }}
        >
            <strong>{label}</strong>
            <p style={{ margin: "8px 0", fontSize: 13 }}>
                Drag and drop a folder or PDF files here.
            </p>
            {accumulatedFiles.size > 0 && !uploading && (
                <div
                    style={{
                        marginTop: 8,
                        fontSize: 12,
                        color: "#666",
                        textAlign: "left",
                    }}
                >
                    <p style={{ margin: "0 0 4px 0" }}>
                        {accumulatedFiles.size} file(s) in queue:
                    </p>
                    <ul
                        style={{
                            margin: 0,
                            paddingLeft: 20,
                            maxHeight: 80,
                            overflow: "auto",
                        }}
                    >
                        {Array.from(accumulatedFiles.keys())
                            .slice(0, 15)
                            .map((name) => (
                                <li key={name}>{name}</li>
                            ))}
                        {accumulatedFiles.size > 15 && (
                            <li>…and {accumulatedFiles.size - 15} more</li>
                        )}
                    </ul>
                </div>
            )}
            {uploading && (
                <p style={{ margin: 8, color: "#07c" }}>Uploading…</p>
            )}
            {result && (
                <div style={{ marginTop: 12, fontSize: 13, textAlign: "left" }}>
                    {result.ok ? (
                        <>
                            <p style={{ color: "green" }}>
                                Done: {result.processed} file(s) processed
                            </p>
                            {result.results?.length ? (
                                <ul
                                    style={{
                                        margin: 0,
                                        paddingLeft: 20,
                                        maxHeight: 120,
                                        overflow: "auto",
                                    }}
                                >
                                    {(
                                        result.results as {
                                            filename?: string
                                            name?: string
                                            ok?: boolean
                                            error?: string
                                        }[]
                                    )
                                        .slice(0, 20)
                                        .map((r, i) => (
                                            <li key={i}>
                                                {String(
                                                    r.filename ??
                                                    r.name ??
                                                    `File ${i + 1}`
                                                )}{" "}
                                                {r.ok !== false
                                                    ? "✓"
                                                    : `✗ ${r.error || ""}`}
                                            </li>
                                        ))}
                                    {(result.results?.length ?? 0) > 20 && (
                                        <li>
                                            …and{" "}
                                            {(result.results?.length ?? 0) - 20}{" "}
                                            more
                                        </li>
                                    )}
                                </ul>
                            ) : null}
                        </>
                    ) : (
                        <p style={{ color: "crimson" }}>{result.error}</p>
                    )}
                </div>
            )}
        </div>
    )
}

// --- Pipeline run ---
const PIPELINE_TIMEOUT_MS = 120000 // 2 min so request does not hang (Framer / CORS)
type RunResult = {
    groups: { snapshot_date: string; supplier: string; items: unknown[] }[]
    reports: {
        snapshot_date: string
        supplier: string
        filename: string
        saved_path: string
        content_base64?: string
    }[]
    requests: {
        snapshot_date: string
        supplier: string
        filename: string
        saved_path: string
        content_base64?: string
    }[]
    emails: {
        snapshot_date: string
        supplier: string
        filename: string
        saved_path: string
        content_base64?: string
    }[]
    evaluations: {
        snapshot_date: string
        supplier: string
        filename: string
        saved_path: string
        content_base64?: string
    }[]
}
type StreamEvent = {
    step: string
    supplier?: string
    count?: number
    filename?: string
    content_base64?: string
    result?: RunResult
    error?: string
}

function PipelineRunBlock({
    apiBase,
    apiToken,
}: {
    apiBase: string
    apiToken: string
}) {
    const [file, setFile] = useState<File | null>(null)
    const [running, setRunning] = useState(false)
    const [statusMessage, setStatusMessage] = useState<string | null>(null)
    const [result, setResult] = useState<RunResult | null>(null)
    const [error, setError] = useState<string | null>(null)

    const onFileChange = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const f = e.target.files?.[0]
            setFile(f || null)
            setResult(null)
            setError(null)
            setStatusMessage(null)
        },
        []
    )

    const runPipeline = useCallback(async () => {
        if (!file) {
            setError("Please select a CSV file.")
            return
        }
        if (!file.name.toLowerCase().endsWith(".csv")) {
            setError("Only CSV files are allowed.")
            return
        }
        setRunning(true)
        setError(null)
        setResult(null)
        setStatusMessage("Starting…")
        const form = new FormData()
        form.append("file", file)
        const controller = new AbortController()
        const timeoutId = setTimeout(
            () => controller.abort(),
            PIPELINE_TIMEOUT_MS
        )
        const url = apiBase
            ? `${apiBase.replace(/\/$/, "")}/api/run/stream`
            : "/api/run/stream"
        try {
            const res = await fetch(url, {
                method: "POST",
                body: form,
                headers: {
                    "X-API-Key": apiToken,
                },
                signal: controller.signal,
            })
            if (!res.ok) {
                clearTimeout(timeoutId)
                const data = (await res.json().catch(() => ({}))) as {
                    detail?: string
                }
                const msg =
                    res.status === 0
                        ? "Request blocked (check apiBaseUrl and CORS)."
                        : data.detail || res.statusText || "Run failed"
                setError(msg)
                setRunning(false)
                setStatusMessage(null)
                return
            }
            const reader = res.body?.getReader()
            const decoder = new TextDecoder()
            let buffer = ""
            if (!reader) {
                setError("Streaming is not supported.")
                setRunning(false)
                setStatusMessage(null)
                return
            }
            const parseOne = (raw: string) => {
                const match = raw.match(/^data:\s*(\S[\s\S]*)/m)
                if (!match) return
                try {
                    const ev = JSON.parse(match[1].trim()) as StreamEvent
                    if (ev.step === "generating_file" && ev.filename) {
                        setStatusMessage(`Generating ${ev.filename}`)
                    }
                    if (
                        ev.step === "file_ready" &&
                        ev.filename &&
                        ev.content_base64
                    ) {
                        setStatusMessage(`Downloading ${ev.filename}`)
                        setTimeout(() => {
                            downloadFile(
                                {
                                    filename: ev.filename!,
                                    content_base64: ev.content_base64!,
                                },
                                (msg) => setError(msg)
                            )
                        }, 0)
                    }
                    if (ev.step === "csv_parsing")
                        setStatusMessage("Parsing CSV…")
                    if (ev.step === "item_grouping")
                        setStatusMessage("Analyzing data…")
                    if (ev.step === "item_grouping_done")
                        setStatusMessage(
                            `${ev.count ?? 0} supplier group(s). Generating documents…`
                        )
                    if (ev.step === "analysis" && ev.supplier)
                        setStatusMessage(
                            `Generating analysis (${ev.supplier})…`
                        )
                    if (ev.step === "pr" && ev.supplier)
                        setStatusMessage(`Generating PR (${ev.supplier})…`)
                    if (ev.step === "email" && ev.supplier)
                        setStatusMessage(
                            `Generating email draft (${ev.supplier})…`
                        )
                    if (ev.step === "complete" && ev.result) {
                        setResult(ev.result)
                        setStatusMessage("Complete")
                    }
                    if (ev.step === "error")
                        setError(ev.error ?? "An error occurred")
                } catch {
                    // ignore
                }
            }
            while (true) {
                const { done, value } = await reader.read()
                if (done) break
                buffer += decoder.decode(value, { stream: true })
                const parts = buffer.split(/\r?\n\r?\n/)
                buffer = parts.pop() ?? ""
                for (const chunk of parts) {
                    if (chunk.trim().startsWith("data:")) parseOne(chunk)
                }
            }
            if (buffer.trim().startsWith("data:")) parseOne(buffer)
            clearTimeout(timeoutId)
        } catch (e) {
            clearTimeout(timeoutId)
            const err = e as Error & { name?: string }
            const msg =
                err.name === "AbortError"
                    ? "Pipeline timed out. Check apiBaseUrl and CORS for your Framer site."
                    : String(e)
            setError(msg)
        } finally {
            setRunning(false)
            setStatusMessage(null)
        }
    }, [file, apiBase])

    return (
        <div
            style={{
                border: "2px solid #07c",
                borderRadius: 8,
                padding: 24,
                marginBottom: 32,
                background: "#f8fcff",
            }}
        >
            <h2 style={{ marginTop: 0 }}>Document Generation</h2>
            <p style={{ color: "#666", marginBottom: 16 }}>
                Upload a stock list CSV and run. Analysis report, PR document,
                and email draft are generated as Word (.docx) and downloaded to
                your browser.
                <br />
                <span style={{ fontSize: 11, color: "#07c", display: "inline-flex", alignItems: "center", gap: 4, marginTop: 8 }}>
                    <span style={{ width: 8, height: 8, background: "#00c853", borderRadius: "50%", display: "inline-block", boxShadow: "0 0 4px #00c853" }}></span>
                    LLMOps Monitoring Active (LangSmith Tracing Enabled)
                </span>
            </p>
            <div
                style={{
                    display: "flex",
                    flexWrap: "wrap",
                    alignItems: "center",
                    gap: 12,
                    marginBottom: 16,
                }}
            >
                <input
                    type="file"
                    accept=".csv"
                    onChange={onFileChange}
                    style={{ fontSize: 14 }}
                />
                <button
                    type="button"
                    onClick={runPipeline}
                    disabled={!file || running}
                    style={{
                        padding: "8px 16px",
                        fontSize: 14,
                        background: file && !running ? "#07c" : "#ccc",
                        color: "#fff",
                        border: "none",
                        borderRadius: 6,
                        cursor: file && !running ? "pointer" : "not-allowed",
                    }}
                >
                    {running ? "Running…" : "Generate Documents"}
                </button>
            </div>
            {result &&
                (result.reports?.length ||
                    result.requests?.length ||
                    result.emails?.length) ? (
                <div
                    style={{
                        marginBottom: 16,
                        padding: 12,
                        background: "#f0f8ff",
                        borderRadius: 6,
                        fontSize: 13,
                    }}
                >
                    <strong>Generated files</strong>
                    <ul
                        style={{
                            margin: "8px 0 0 0",
                            paddingLeft: 20,
                            listStyle: "none",
                        }}
                    >
                        {[
                            ...(result.reports ?? []),
                            ...(result.requests ?? []),
                            ...(result.emails ?? []),
                            ...(result.evaluations ?? []),
                        ].map((r, i) => (
                            <li
                                key={`${r.filename}-${i}`}
                                style={{
                                    marginBottom: 6,
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 8,
                                }}
                            >
                                <span style={{ wordBreak: "break-all" }}>
                                    {r.filename}
                                </span>
                                <button
                                    type="button"
                                    onClick={() =>
                                        downloadFile(r, (msg) => setError(msg))
                                    }
                                    style={{
                                        flexShrink: 0,
                                        padding: "4px 10px",
                                        fontSize: 12,
                                        background: "#07c",
                                        color: "#fff",
                                        border: "none",
                                        borderRadius: 4,
                                        cursor: "pointer",
                                    }}
                                >
                                    Download
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            ) : null}
            {running && statusMessage && (
                <div
                    style={{
                        padding: "12px 16px",
                        marginBottom: 16,
                        background: "#e8f4fc",
                        borderRadius: 6,
                        fontSize: 14,
                        fontWeight: 500,
                    }}
                >
                    {statusMessage}
                </div>
            )}
            {file && !running && (
                <p style={{ fontSize: 13, color: "#666", marginBottom: 8 }}>
                    Selected: <strong>{file.name}</strong>
                </p>
            )}
            {error && (
                <p style={{ color: "crimson", marginBottom: 8 }}>{error}</p>
            )}
            {result && (
                <div style={{ fontSize: 13, marginTop: 8 }}>
                    <p style={{ color: "green", fontWeight: "bold" }}>
                        Done: {result.groups?.length ?? 0} supplier group(s)
                        processed.
                    </p>
                </div>
            )}
        </div>
    )
}

const DEFAULT_API_BASE = "https://purchasing-automation-531560336160.us-central1.run.app"

export default function PurchasingAutomationFramer({
    apiBaseUrl = "",
    apiAccessToken = "",
}: {
    apiBaseUrl?: string
    apiAccessToken?: string
}) {
    const apiBase = (apiBaseUrl || DEFAULT_API_BASE).replace(/\/$/, "")
    const apiToken = apiAccessToken
    return (
        <div
            style={{
                maxWidth: 640,
                margin: "0 auto",
                padding: 24,
                paddingBottom: 48,
                fontFamily: "system-ui, sans-serif",
                minHeight: "max(100vh, 2000px)",
            }}
        >
            <h1>Purchasing Automation</h1>
            <br></br>
            <PipelineRunBlock apiBase={apiBase} apiToken={apiToken} />
            <h2>Knowledge Base & Reference Library</h2>
            <p style={{ color: "#666", marginBottom: 24 }}>
                Provide the AI with previous purchase records and document
                samples to ensure tailored and high-quality outputs. Drag and
                drop a folder or PDF files (Chrome, Edge). PDFs inside a folder
                are collected automatically.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <FolderDropZone
                    apiBase={apiBase}
                    apiToken={apiToken}
                    ingestType="supplier-history"
                    label="Supplier History"
                />
                <FolderDropZone
                    apiBase={apiBase}
                    apiToken={apiToken}
                    ingestType="item-history"
                    label="Item History"
                />
                <FolderDropZone
                    apiBase={apiBase}
                    apiToken={apiToken}
                    ingestType="analysis-examples"
                    label="Analysis Examples"
                />
                <FolderDropZone
                    apiBase={apiBase}
                    apiToken={apiToken}
                    ingestType="request-examples"
                    label="Request Examples (PR)"
                />
                <FolderDropZone
                    apiBase={apiBase}
                    apiToken={apiToken}
                    ingestType="email-examples"
                    label="Email Examples"
                />
            </div>
        </div>
    )
}
addPropertyControls(PurchasingAutomationFramer, {
    apiBaseUrl: {
        type: ControlType.String,
        title: "API Base URL",
        defaultValue: DEFAULT_API_BASE,
    },
    apiAccessToken: {
        type: ControlType.String,
        title: "API Access Token",
        defaultValue: "",
    },
})
