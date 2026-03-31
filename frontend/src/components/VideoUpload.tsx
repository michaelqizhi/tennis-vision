"use client";

import { useCallback, useState, useRef } from "react";

interface VideoUploadProps {
  onUploadComplete: (jobId: string) => void;
  disabled?: boolean;
}

export default function VideoUpload({
  onUploadComplete,
  disabled = false,
}: VideoUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const ACCEPTED_TYPES = [".mp4", ".avi", ".mov", ".mkv", ".webm"];

  const validateFile = (file: File): string | null => {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_TYPES.includes(ext)) {
      return `Unsupported file type: ${ext}. Accepted: ${ACCEPTED_TYPES.join(", ")}`;
    }
    if (file.size > 500 * 1024 * 1024) {
      return "File too large. Maximum size is 500 MB.";
    }
    return null;
  };

  const handleUpload = useCallback(
    async (file: File) => {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }

      setError(null);
      setFileName(file.name);
      setUploading(true);

      try {
        const { uploadVideo } = await import("@/lib/api");
        const response = await uploadVideo(file);
        onUploadComplete(response.job_id);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Upload failed. Please try again."
        );
      } finally {
        setUploading(false);
      }
    },
    [onUploadComplete]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (disabled || uploading) return;
      const file = e.dataTransfer.files[0];
      if (file) handleUpload(file);
    },
    [disabled, uploading, handleUpload]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (disabled || uploading) return;
      const file = e.target.files?.[0];
      if (file) handleUpload(file);
    },
    [disabled, uploading, handleUpload]
  );

  return (
    <div className="w-full max-w-xl mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer
          ${dragActive ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10" : "border-[var(--color-card-border)] hover:border-[var(--color-muted)]"}
          ${disabled || uploading ? "opacity-50 cursor-not-allowed" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled && !uploading) setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => {
          if (!disabled && !uploading) inputRef.current?.click();
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          onChange={handleChange}
          className="hidden"
        />

        <div className="space-y-3">
          {/* Upload icon */}
          <svg
            className="mx-auto h-12 w-12 text-[var(--color-muted)]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
            />
          </svg>

          {uploading ? (
            <div>
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--color-accent)] mx-auto mb-2" />
              <p className="text-sm text-[var(--color-muted)]">
                Uploading {fileName}...
              </p>
            </div>
          ) : (
            <>
              <p className="text-lg font-medium">
                Drop your tennis video here
              </p>
              <p className="text-sm text-[var(--color-muted)]">
                or click to browse — MP4, AVI, MOV, MKV, WebM (max 500 MB)
              </p>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-3 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}
