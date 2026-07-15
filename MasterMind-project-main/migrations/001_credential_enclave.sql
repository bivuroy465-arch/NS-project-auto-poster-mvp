-- ============================================================================
-- Migration 001: Credential Enclave schema (PostgreSQL / Neon)
-- ============================================================================
-- Stores AES-256-GCM sealed credentials per user (BYOK model).
-- Plaintext NEVER touches this database: only nonce + ciphertext + tag.
-- The master key lives exclusively in the ENCLAVE_MASTER_KEY env var of the
-- processing fleet; the database alone is cryptographically useless.
--
-- Applied automatically by src/enclave/storage.py::CredentialRepository.migrate()
-- (idempotent — safe to run on every cold start).
-- ============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS enclave_users (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id   TEXT        NOT NULL UNIQUE,          -- JWT `sub` claim
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    disabled_at   TIMESTAMPTZ                            -- soft-disable, keys retained
);

CREATE TABLE IF NOT EXISTS enclave_credentials (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL REFERENCES enclave_users(id) ON DELETE CASCADE,

    -- Logical name, e.g. 'OPENAI_API_KEY'. Doubles as the GCM AAD:
    -- a ciphertext cannot be moved to a different name without failing auth.
    name            TEXT        NOT NULL,

    -- AES-256-GCM material. nonce is 12 bytes; ciphertext includes the
    -- 16-byte GCM tag appended by the encryptor.
    nonce           BYTEA       NOT NULL CHECK (octet_length(nonce) = 12),
    ciphertext      BYTEA       NOT NULL CHECK (octet_length(ciphertext) >= 16),

    -- Key-rotation bookkeeping: which master-key generation sealed this row.
    key_version     INTEGER     NOT NULL DEFAULT 1,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Audit trail (timestamps only — never values, never partial values).
    last_used_at    TIMESTAMPTZ,
    use_count       BIGINT      NOT NULL DEFAULT 0,

    CONSTRAINT enclave_credentials_user_name_uniq UNIQUE (user_id, name)
);

-- Hot path: fleet worker resolves (user, name) → blob on every job.
CREATE INDEX IF NOT EXISTS idx_enclave_credentials_user_name
    ON enclave_credentials (user_id, name);

-- Rotation path: find all rows sealed under an old key generation.
CREATE INDEX IF NOT EXISTS idx_enclave_credentials_key_version
    ON enclave_credentials (key_version);

-- Append-only audit log of enclave operations (SOC 2 evidence trail).
CREATE TABLE IF NOT EXISTS enclave_audit_log (
    id          BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     UUID,                                    -- nullable: system ops
    action      TEXT        NOT NULL CHECK (action IN
                    ('store', 'inject', 'delete', 'rotate', 'decrypt_failure')),
    credential  TEXT,                                    -- logical name only
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    detail      TEXT                                     -- NEVER secret material
);

CREATE INDEX IF NOT EXISTS idx_enclave_audit_log_user_at
    ON enclave_audit_log (user_id, at DESC);

COMMIT;
