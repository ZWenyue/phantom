#!/usr/bin/env bash
# Same as b/run_contact_retarget.sh (export objects/hands, then intent→overlay).
exec "$(cd "$(dirname "$0")" && pwd)/run_contact_retarget.sh" "$@"
