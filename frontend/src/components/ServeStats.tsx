"use client";

import type { ServeStats as ServeStatsType } from "@/lib/api";

interface ServeStatsProps {
  stats: ServeStatsType;
}

export default function ServeStats({ stats }: ServeStatsProps) {
  return (
    <div className="bg-[var(--color-card)] border border-[var(--color-card-border)] rounded-xl p-4 space-y-6">
      <h3 className="text-lg font-semibold">Serve Statistics</h3>

      {stats.total_serves === 0 ? (
        <p className="text-[var(--color-muted)] text-sm text-center py-4">
          No serves detected.
        </p>
      ) : (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatCard label="Total Serves" value={stats.total_serves} />
            <StatCard
              label="1st Serve %"
              value={`${stats.first_serve_pct.toFixed(0)}%`}
              color={stats.first_serve_pct >= 60 ? "green" : stats.first_serve_pct >= 40 ? "yellow" : "red"}
            />
            <StatCard label="Double Faults" value={stats.double_faults} color={stats.double_faults > 5 ? "red" : "default"} />
            <StatCard label="Aces" value={stats.aces} color={stats.aces > 0 ? "green" : "default"} />
          </div>

          {/* Serve breakdown */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-[var(--color-muted)]">
              Serve Breakdown
            </h4>

            {/* 1st serve bar */}
            <ServeBar
              label="1st Serves"
              total={stats.first_serves}
              made={stats.first_serve_in}
              faults={stats.first_serve_faults}
            />

            {/* 2nd serve bar */}
            <ServeBar
              label="2nd Serves"
              total={stats.second_serves}
              made={stats.second_serve_in}
              faults={stats.second_serves - stats.second_serve_in}
            />
          </div>

          {/* Visual stats grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-[var(--color-bg)] rounded-lg p-3">
              <p className="text-xs text-[var(--color-muted)]">1st Serve In</p>
              <p className="text-2xl font-bold text-[var(--color-accent)]">
                {stats.first_serve_in}
                <span className="text-sm font-normal text-[var(--color-muted)]">
                  /{stats.first_serves}
                </span>
              </p>
            </div>
            <div className="bg-[var(--color-bg)] rounded-lg p-3">
              <p className="text-xs text-[var(--color-muted)]">2nd Serve %</p>
              <p className="text-2xl font-bold">
                {stats.second_serve_pct.toFixed(0)}%
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
  color = "default",
}: {
  label: string;
  value: string | number;
  color?: "default" | "green" | "yellow" | "red";
}) {
  const colorClass =
    color === "green"
      ? "text-green-400"
      : color === "yellow"
      ? "text-yellow-400"
      : color === "red"
      ? "text-red-400"
      : "";

  return (
    <div className="bg-[var(--color-bg)] rounded-lg p-3">
      <p className="text-xs text-[var(--color-muted)]">{label}</p>
      <p className={`text-xl font-bold mt-1 ${colorClass}`}>{value}</p>
    </div>
  );
}

function ServeBar({
  label,
  total,
  made,
  faults,
}: {
  label: string;
  total: number;
  made: number;
  faults: number;
}) {
  if (total === 0) return null;
  const madePct = (made / total) * 100;
  const faultPct = (faults / total) * 100;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span>{label}</span>
        <span className="text-[var(--color-muted)]">
          {made} in / {faults} fault{faults !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="h-4 bg-[var(--color-bg)] rounded-full overflow-hidden flex">
        <div
          className="h-full bg-[var(--color-accent)] transition-all"
          style={{ width: `${madePct}%` }}
        />
        <div
          className="h-full bg-red-500/70 transition-all"
          style={{ width: `${faultPct}%` }}
        />
      </div>
    </div>
  );
}
