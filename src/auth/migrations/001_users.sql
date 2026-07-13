-- Users table for authentication and authorization
-- Used by UserStore (db feature) and init_schema in src/db/mod.rs
--
-- Schema mirrors the inline CREATE TABLE in DbPool::init_schema to provide
-- a migration file source of truth (R-persistence-002).

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    permissions TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
