###Sync data between the backend and frontend.

```plaintext
/v1/sync
```

Method: `Get`

Type: `text/events-stream`

##### JSON Example:

For full update:

```json5
{
  "id": 1,
  "workspace": 1,
  "event": "sync",
  "agents": [
    {
      "id": 1,
      "name": "name_string",
      "description": "desc_string",
      "sub_agent": [
        2
      ]
    },
    {
      "id": 2,
      "name": "sync",
      "description": "desc_string",
      "sub_agent": [
      ]
    }
  ]
}
```

For incremental update:

```json5
{
  "id": 2,
  "workspace": 1,
  "event": "event_string",
  "agents": [
    {
      "id": 1,
      "name": "new_name_string"
    }
  ]
}
```

### Get streaming responses of  a agent.

```plaintext
/v1/message
```

Method: `Get`

Type: `text/events-stream`

##### Parameters

| Key   | Type | Info                                         |
|-------|------|----------------------------------------------|
| id    | int  | The number id of the target agent            |
| block | int  | The number id of the requested message block |

##### Note

If `block` exists, the backend will return all the message blocks whose number id is
***greater or equal than `block`***.

##### JSON  Streaming Example

The whole block:

```json
{
  "type": "token",
  "agent_id": 1,
  "content": "Ultramarine is out of blue grass but deeper than blue; Ice is frozen water but colder than water. A length of straight wood, when baked into a wheel, will never be straight again, even if dried, because of the baking. Therefore, wood will remain straight by the ink line, metal tools get sharper on the grindstones., and Wisemen ,though learned, make self-reflections everyday, so they are wise enough not to behave wrong. I "
}
```

Single plain token:

```json
{
  "type": "token",
  "agent_id": 1,
  "content": "once"
}
```

Tool calling header:

```json
{
  "type": "tool_call",
  "agent_id": 1,
  "action": "file.write",
  "description": "Write File",
  "additional": "/tmp/hello_world.txt"
}
```

Tool calling body:

```json
{
  "type": "tool_call",
  "agent_id": 1,
  "content": "import "
}
```

Tool calling result:

```json
{
  "type": "tool_result",
  "agent_id": 1,
  "result": "The operation is successful"
}
```