-- ===========================================================================
-- P10 - Predictive Retail & Economic Data Dashboard
-- Supabase / PostgreSQL schema with Row-Level Security (tenant isolation)
--
-- Apply with the Supabase CLI:
--     supabase db push
-- or paste into the SQL editor.
-- ===========================================================================

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- Tenants
-- ---------------------------------------------------------------------------
create table if not exists public.tenants (
    tenant_id     text primary key,
    company_name  text not null,
    api_key_hash  text not null unique,
    created_at    timestamptz not null default now()
);

comment on table public.tenants is 'One row per enterprise customer (retailer / financial institution).';

-- Maps Supabase Auth users to a tenant. A user belongs to exactly one tenant in
-- this MVP; the unique constraint documents that assumption explicitly.
create table if not exists public.tenant_members (
    user_id    uuid not null references auth.users(id) on delete cascade,
    tenant_id  text not null references public.tenants(tenant_id) on delete cascade,
    role       text not null default 'analyst'
               check (role in ('analyst', 'executive', 'admin')),
    created_at timestamptz not null default now(),
    primary key (user_id, tenant_id)
);

-- ---------------------------------------------------------------------------
-- Economic / market data
-- ---------------------------------------------------------------------------
create table if not exists public.economic_data (
    record_id     bigint generated always as identity primary key,
    tenant_id     text not null references public.tenants(tenant_id) on delete cascade,
    asset_name    text not null,
    date_logged   date not null,
    closing_price numeric(18, 6) not null check (closing_price > 0),
    volume        numeric(20, 4) not null default 0 check (volume >= 0),
    zar_usd       numeric(12, 6),
    cpi           numeric(12, 4),
    fuel_price    numeric(12, 4),
    repo_rate     numeric(8, 4),
    source        text not null default 'upload',
    ingested_at   timestamptz not null default now(),
    constraint ux_economic_data_key unique (tenant_id, asset_name, date_logged)
);

create index if not exists ix_economic_data_lookup
    on public.economic_data (tenant_id, asset_name, date_logged desc);

-- ---------------------------------------------------------------------------
-- Model telemetry
-- ---------------------------------------------------------------------------
create table if not exists public.model_metrics (
    metric_id    bigint generated always as identity primary key,
    tenant_id    text not null references public.tenants(tenant_id) on delete cascade,
    asset_name   text not null,
    algorithm    text not null,
    trained_at   timestamptz not null default now(),
    rows_trained integer not null,
    mae          double precision not null,
    rmse         double precision not null,
    mape         double precision not null,
    directional_accuracy double precision not null
);

create index if not exists ix_model_metrics_lookup
    on public.model_metrics (tenant_id, asset_name, trained_at desc);

-- ---------------------------------------------------------------------------
-- Audit trail
-- ---------------------------------------------------------------------------
create table if not exists public.audit_log (
    audit_id   bigint generated always as identity primary key,
    tenant_id  text not null,
    action     text not null,
    detail     text not null,
    created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- Tenant resolution helper
-- ---------------------------------------------------------------------------
create or replace function public.current_tenant_id()
returns text
language sql
stable
security definer
set search_path = public
as $$
    select tenant_id
    from public.tenant_members
    where user_id = auth.uid()
    limit 1;
$$;

comment on function public.current_tenant_id() is
    'Resolves the tenant of the authenticated Supabase user; used by every RLS policy.';

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
alter table public.tenants         enable row level security;
alter table public.tenant_members  enable row level security;
alter table public.economic_data   enable row level security;
alter table public.model_metrics   enable row level security;
alter table public.audit_log       enable row level security;

-- Force RLS for table owners too (defence in depth).
alter table public.economic_data force row level security;
alter table public.model_metrics force row level security;

drop policy if exists tenants_read on public.tenants;
create policy tenants_read on public.tenants
    for select to authenticated
    using (tenant_id = public.current_tenant_id());

drop policy if exists members_read on public.tenant_members;
create policy members_read on public.tenant_members
    for select to authenticated
    using (user_id = auth.uid());

-- economic_data: a tenant can only ever touch its own rows.
drop policy if exists economic_data_select on public.economic_data;
create policy economic_data_select on public.economic_data
    for select to authenticated
    using (tenant_id = public.current_tenant_id());

drop policy if exists economic_data_insert on public.economic_data;
create policy economic_data_insert on public.economic_data
    for insert to authenticated
    with check (tenant_id = public.current_tenant_id());

drop policy if exists economic_data_update on public.economic_data;
create policy economic_data_update on public.economic_data
    for update to authenticated
    using (tenant_id = public.current_tenant_id())
    with check (tenant_id = public.current_tenant_id());

drop policy if exists economic_data_delete on public.economic_data;
create policy economic_data_delete on public.economic_data
    for delete to authenticated
    using (tenant_id = public.current_tenant_id());

drop policy if exists model_metrics_read on public.model_metrics;
create policy model_metrics_read on public.model_metrics
    for select to authenticated
    using (tenant_id = public.current_tenant_id());

drop policy if exists model_metrics_write on public.model_metrics;
create policy model_metrics_write on public.model_metrics
    for insert to authenticated
    with check (tenant_id = public.current_tenant_id());

drop policy if exists audit_log_read on public.audit_log;
create policy audit_log_read on public.audit_log
    for select to authenticated
    using (tenant_id = public.current_tenant_id());

drop policy if exists audit_log_write on public.audit_log;
create policy audit_log_write on public.audit_log
    for insert to authenticated
    with check (tenant_id = public.current_tenant_id());

-- The service role used by the FastAPI backend bypasses RLS, but every query in
-- app/db/repository.py still carries an explicit tenant predicate.
