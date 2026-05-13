---
dimension: performance
weight: 0
threshold:
  pass: 80
  warn: 70
metrics:
  - name: web_route_performance_smoke
    command: npm run test:performance 2>&1
    pattern: "✅"
    tier: deep
    execution_scope: ci
    gate: advisory
    kind: holistic
    analysis: dynamic
    stability: noisy
    evidence_type: test
    scope: [web]
    run_when_changed:
      - src/app/**
      - src/client/**
      - src/app/styles/**
      - e2e/**
      - scripts/fitness/check-performance-smoke.mjs
      - scripts/page-snapshot-lib.mjs
    description: "关键 workspace / kanban / traces / session detail 路由的导航、FCP、CSS 体积与 long task smoke"

  - name: sqlite_wal_mode_guard
    command: rg -q 'journal_mode = WAL' src/core/db/sqlite.ts && echo 'sqlite_wal_mode_ok'
    pattern: "sqlite_wal_mode_ok"
    tier: normal
    execution_scope: ci
    gate: soft
    kind: atomic
    analysis: static
    evidence_type: command
    scope: [web]
    run_when_changed:
      - src/core/db/sqlite.ts
    description: "本地 Node/SQLite 后端继续启用 WAL，避免回退到较差的并发读性能"
---

# Performance Runtime Evidence

这份文件只负责运行时性能证据，不承担 design system 或一般静态代码质量的职责。

## 当前覆盖

- `web_route_performance_smoke`
    - 命令：`npm run test:performance`
    - 环境：`ci`
    - 语义：`advisory`，失败会暴露回退，但不会替代发布前的真实产线或 staging 预算
- `sqlite_wal_mode_guard`
    - 命令：静态检查 `src/core/db/sqlite.ts` 是否继续启用 `journal_mode = WAL`
    - 环境：`ci`
    - 语义：`soft`

## 边界

- 这里的 smoke 用于发现明显性能回退，不声明自己是 production latency 的事实来源。
- `performance` 与 `observability` 分离：有 tracing 或错误信号，不等于性能达标。
