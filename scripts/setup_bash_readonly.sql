-- Phase F: Create read-only PostgreSQL role for bash agent sandbox.
-- Run against the epic_sim database as a superuser.
--
-- Usage: psql -U epic_sim -d epic_sim -f scripts/setup_bash_readonly.sql

-- Create role (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bash_readonly') THEN
        CREATE ROLE bash_readonly WITH LOGIN PASSWORD 'readonly_password';
    END IF;
END $$;

-- Grant minimal access
GRANT CONNECT ON DATABASE epic_sim TO bash_readonly;
GRANT USAGE ON SCHEMA public TO bash_readonly;

-- Grant SELECT on exactly 9 clinical data tables (whitelist)
GRANT SELECT ON longitudinal_patients TO bash_readonly;
GRANT SELECT ON longitudinal_encounters TO bash_readonly;
GRANT SELECT ON encounter_ehr_sections TO bash_readonly;
GRANT SELECT ON diagnoses TO bash_readonly;
GRANT SELECT ON clinical_findings TO bash_readonly;
GRANT SELECT ON diagnosis_findings TO bash_readonly;
GRANT SELECT ON fact_cards TO bash_readonly;
GRANT SELECT ON imaging_orders TO bash_readonly;
GRANT SELECT ON terminology_codes TO bash_readonly;

-- Revoke any default public grants that might leak access
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;

-- Re-grant to the main epic_sim user (owner keeps access regardless)
GRANT ALL ON ALL TABLES IN SCHEMA public TO epic_sim;
