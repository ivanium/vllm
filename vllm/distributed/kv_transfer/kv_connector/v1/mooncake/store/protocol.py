# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-format constants for the MooncakeStoreConnector ZMQ channels.

Two channels share these constants:

1. Admin channel (scheduler -> worker rank 0, REQ/REP), served by
   ``StoreAdminServer``:

      Request: [RESET_MSG]
        Drains the send queue, then runs ``store.remove_all(force=True)``.
      Response: [RESP_OK] or [RESP_ERR]

2. Lookup channel (scheduler -> lookup subprocess, DEALER/REP), served by
   ``_lookup_server_main``:

      Request: [b""] [wire_id: utf-8] [msg_type: bytes]
               [token_len: u32 big-endian]
               [hash_len: u16 big-endian — byte length of each fixed-size
                block hash, 0 when there are no hashes]
               [raw block hashes concatenated back-to-back]
               [session_id: utf-8, may be empty]

        The leading empty frame is the REQ/REP envelope delimiter that a
        DEALER socket must add by hand; the REP server never sees it.

        msg_type == LOOKUP_MSG: prefix-cache hit query. ``wire_id`` is
          ``<seq>|<request_id>`` — unique per lookup attempt, so a reply
          from a discarded attempt can't be matched to a later request
          reusing the same request id.
        msg_type == RECORD_BP_MSG: record a session breakpoint
          (fire-and-forget; ``wire_id`` is empty so the client drops the
          reply).

      Response (REP application frames): [wire_id] [hit_count: u32
        big-endian]. On the DEALER side the reply surfaces with the
        envelope delimiter prepended: [b""] [wire_id] [hit_count].

Message types are named bytes tags (not numeric sentinels that alias the
data field) so the protocol stays self-describing and extensible: adding a
new command requires only a new tag and a new dispatch branch. Mirrors the
named-tag convention used by the NIXL connector (see
``vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py``).
"""

# Request message-type tags.
LOOKUP_MSG: bytes = b"lookup"
RECORD_BP_MSG: bytes = b"record_bp"
RESET_MSG: bytes = b"reset"

# Single-byte response status codes for admin commands.
RESP_OK: bytes = b"\x01"
RESP_ERR: bytes = b"\x00"
