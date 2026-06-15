// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '200mb' }));
app.use(express.urlencoded({ limit: '200mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = true; // Set to true to show reasoning with <think> tags

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'glm-5': 'z-ai/glm5',
  'kimi-k2.6': 'moonshotai/kimi-k2.6',
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3.2',
  'gpt-oss-120b': 'openai/gpt-oss-120b',
  'minimax-m2.7': 'minimaxai/minimax-m2.7',
  'glm-5.1': 'z-ai/glm-5.1',
  'glm-4.7': 'z-ai/glm4.7',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'qwen-3.5': 'qwen/qwen3.5-397b-a17b',
  'nemotron-3-super': 'nvidia/nemotron-3-super-120b-a12b',
  'mistral-large-3': 'mistralai/mistral-large-3-675b-instruct-2512',
  'step-3.7': 'stepfun-ai/step-3.7-flash',
  'nemotron-3-ultra': 'nvidia/nemotron-3-ultra-550b-a55b',
  'minimax-m3': 'minimaxai/minimax-m3'
};

// These models use a different thinking param style (thinking/reasoning_effort instead of enable_thinking)
const thinking_models = ["z-ai/glm-5.1", "qwen/qwen3.5-397b-a17b", "stepfun-ai/step-3.7-flash", "nvidia/nemotron-3-ultra-550b-a55b"];

// These models always think on their own — don't send any chat_template_kwargs
const auto_think_models = ["mistralai/mistral-large-3-675b-instruct-2512", "stepfun-ai/step-3.7-flash"];

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    tip: 'Append -no-think to any model name to disable thinking (e.g. minimax-m2.7-no-think)'
  });
});

// List models endpoint (OpenAI compatible)
// Exposes both normal and -no-think variants for every model
app.get('/v1/models', (req, res) => {
  const models = [];
  Object.keys(MODEL_MAPPING).forEach(model => {
    models.push({ id: model,               object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' });
    models.push({ id: `${model}-no-think`, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' });
  });

  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;

    // ── Detect and strip -no-think suffix ──────────────────────────────────
    const noThinkRequested = model.endsWith('-no-think');
    const cleanModel = noThinkRequested ? model.slice(0, -'-no-think'.length) : model;
    // ───────────────────────────────────────────────────────────────────────

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[cleanModel];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: cleanModel,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(r => {
          if (r.status >= 200 && r.status < 300) nimModel = cleanModel;
        });
      } catch (e) {}

      if (!nimModel) {
        const modelLower = cleanModel.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // ── Build chat_template_kwargs based on thinking preference ────────────
    let chat_template_kwargs;

    if (noThinkRequested) {
      // User explicitly wants NO thinking
      chat_template_kwargs = {};
    } else {
      // Default: thinking ON
      if (thinking_models.includes(nimModel)) {
        chat_template_kwargs = { thinking: true, reasoning_effort: "high" };
      } else {
        chat_template_kwargs = { enable_thinking: true, clear_thinking: false, thinking_mode: "enabled" };
      }
    }
    // ───────────────────────────────────────────────────────────────────────

    // Build NIM request
    let nimRequest;
    if (auto_think_models.includes(nimModel)) {
      // These models manage thinking themselves — never send chat_template_kwargs
      nimRequest = {
        model: nimModel,
        messages,
        temperature: temperature || 0.6,
        max_tokens: max_tokens || 50000,
        stream: stream || false
      };
    } else {
      nimRequest = {
        model: nimModel,
        messages,
        temperature: temperature || 0.6,
        max_tokens: max_tokens || 50000,
        chat_template_kwargs,
        stream: stream || false
      };
    }

    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;

                if (SHOW_REASONING && !noThinkRequested && !thinking_models.includes(nimModel)) {
                  let combinedContent = '';

                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }

                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }

                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  data.choices[0].delta.content = content || '';
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model, // Echo back the original model name (including -no-think if used)
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && !noThinkRequested && choice.message?.reasoning_content && !thinking_models.includes(nimModel)) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);

    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Tip: append -no-think to any model name to disable thinking`);
});