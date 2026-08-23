import type { AgentSession } from '@waku/client'
import { translate, type AppLocale } from './i18n'

export const RENOA_WEB_REPLAY_ERROR_KEY = 'errors.renoa_web_replay_unsupported'

export function sessionRequiresDesktopRenoaReplay(
  session: Pick<AgentSession, 'provider'>,
): boolean {
  return session.provider === 'renoa'
}

export function renoaWebRuntimeError(
  session: Pick<AgentSession, 'provider'>,
  locale: AppLocale,
): string | null {
  if (!sessionRequiresDesktopRenoaReplay(session)) return null
  return translate(locale, RENOA_WEB_REPLAY_ERROR_KEY)
}
