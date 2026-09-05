WITH activity AS (
    SELECT DISTINCT
        user_id,
        DATE_TRUNC('week', impression_time)::DATE AS activity_week
    FROM candidates
), eligible AS (
    SELECT *
    FROM activity
    WHERE (SELECT COUNT(DISTINCT activity_week) FROM activity) >= 2
), cohort AS (
    SELECT user_id, MIN(activity_week) AS cohort_week
    FROM eligible
    GROUP BY user_id
), cohort_sizes AS (
    SELECT cohort_week, COUNT(*)::BIGINT AS cohort_users
    FROM cohort
    GROUP BY cohort_week
), retained AS (
    SELECT
        cohort.cohort_week,
        DATE_DIFF('week', cohort.cohort_week, eligible.activity_week)::BIGINT AS week_offset,
        COUNT(DISTINCT eligible.user_id)::BIGINT AS active_users
    FROM eligible
    INNER JOIN cohort USING (user_id)
    GROUP BY cohort.cohort_week, week_offset
)
SELECT
    CAST(retained.cohort_week AS VARCHAR) AS cohort_week,
    retained.week_offset,
    cohort_sizes.cohort_users,
    retained.active_users,
    retained.active_users::DOUBLE / NULLIF(cohort_sizes.cohort_users, 0) AS retention_rate
FROM retained
INNER JOIN cohort_sizes USING (cohort_week)
ORDER BY cohort_week, week_offset;
