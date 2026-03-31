"use client";

import type { RallyStats as RallyStatsType, Rally } from "@/lib/api";

interface RallyStatsProps {
  stats: RallyStatsType;
  rallies: Rally[];
}

export default function RallyStats({ stats, rallies }: RallyStatsProps) {
  const maxShots = Math.max(...(stats.shots_per_rally.length > 0 ? stats.shots_per_rally : [0]));

  return (
    <div className="bg-[var(--color-card)] border border-[var(--color-card-border)] rounded-xl p-4 space-y-6">
      <h3 className="text-lg font-semibold">Rally Statistics</h3>

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Rallies" value={stats.rally_count} />
        <StatCard label="Total Shots" value={stats.total_shots} />
        <StatCard label="Avg Rally Length" value={`${stats.avg_rally_length.toFixed(1)}s`} />
        <StatCard label="Avg Shots/Rally" value={stats.avg_shots_per_rally.toFixed(1)} />
      </div>

      {/* Additional stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <StatCard label="Longest Rally" value={`${stats.longest_rally_duration.toFixed(1)}s`} sub={`${stats.longest_rally_shots} shots`} />
        <StatCard label="Shortest Rally" value={`${stats.shortest_rally_duration.toFixed(1)}s`} sub={`${stats.shortest_rally_shots} shots`} />
        <StatCard label="Median Duration" value={`${stats.median_rally_duration.toFixed(1)}s`} sub={`${stats.median_shots_per_rally.toFixed(1)} shots`} />
      </div>

      {/* Shot distribution chart */}
      {stats.shots_per_rally.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-[var(--color-muted)] mb-3">
            Shots per Rally
          </h4>
          <div className="flex items-end gap-1 h-24">
            {stats.shots_per_rally.map((shots, i) => (
              <div
                key={i}
                className="flex-1 bg-[var(--color-accent)]/70 hover:bg-[var(--color-accent)] rounded-t transition-colors relative group min-w-[4px]"
                style={{
                  height: `${maxShots > 0 ? (shots / maxShots) * 100 : 0}%`,
                }}
              >
                <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-xs text-[var(--color-muted)] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                  R{i + 1}: {shots}
                </span>
              </div>
            ))}
          </div>
          <div className="flex justify-between text-xs text-[var(--color-muted)] mt-1">
            <span>Rally 1</span>
            <span>Rally {stats.shots_per_rally.length}</span>
          </div>
        </div>
      )}

      {/* Rally table */}
      {rallies.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-[var(--color-muted)] mb-2">
            Rally Details
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[var(--color-muted)] border-b border-[var(--color-card-border)]">
                  <th className="pb-2 pr-4">#</th>
                  <th className="pb-2 pr-4">Start</th>
                  <th className="pb-2 pr-4">Duration</th>
                  <th className="pb-2">Shots</th>
                </tr>
              </thead>
              <tbody>
                {rallies.map((rally) => (
                  <tr
                    key={rally.rally_id}
                    className="border-b border-[var(--color-card-border)]/50 hover:bg-white/5"
                  >
                    <td className="py-1.5 pr-4 text-[var(--color-muted)]">
                      {rally.rally_id}
                    </td>
                    <td className="py-1.5 pr-4">
                      {formatTime(rally.start_time)}
                    </td>
                    <td className="py-1.5 pr-4">
                      {rally.duration.toFixed(1)}s
                    </td>
                    <td className="py-1.5">{rally.shot_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="bg-[var(--color-bg)] rounded-lg p-3">
      <p className="text-xs text-[var(--color-muted)]">{label}</p>
      <p className="text-xl font-bold mt-1">{value}</p>
      {sub && <p className="text-xs text-[var(--color-muted)]">{sub}</p>}
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
