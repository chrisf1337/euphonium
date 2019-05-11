use lazy_static::lazy_static;
use std::collections::HashSet;

pub type Span = (usize, Tok, usize);
pub type Result = std::result::Result<Span, LexError>;

lazy_static! {
    static ref SYMBOL_START_CHARS: HashSet<char> = {
        let mut set = HashSet::new();
        set.insert('(');
        set.insert(')');
        set.insert('{');
        set.insert('}');
        set.insert('[');
        set.insert(']');
        set.insert(':');
        set.insert(';');
        set.insert(',');
        set.insert('=');
        set.insert('+');
        set.insert('-');
        set.insert('*');
        set.insert('/');
        set.insert('&');
        set.insert('|');
        set.insert('!');
        set.insert('.');
        set.insert('%');
        set
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tok {
    Identifier(String),
    Number(usize),
    String(String),
    Type,
    Equals,
    DoubleEquals,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Colon,
    Semicolon,
    Comma,
    Ampersand,
    And,
    Pipe,
    Or,
    Plus,
    Minus,
    Asterisk,
    Slash,
    Let,
    Mut,
    ThinArrow,
    ThickArrow,
    Fn,
    Not,
    NotEquals,
    True,
    False,
    Underscore,
    Period,
    DoublePeriod,
    If,
    Else,
    Percent,
    PercentLBrace,
    For,
    In,
    While,
    Break,
    Continue,
}

impl std::fmt::Display for Tok {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Tok::Identifier(s) => write!(f, "{}", s),
            Tok::Number(n) => write!(f, "{}", n),
            Tok::String(s) => write!(f, "{}", s),
            Tok::Type => write!(f, "type"),
            Tok::Equals => write!(f, "="),
            Tok::DoubleEquals => write!(f, "=="),
            Tok::LParen => write!(f, "("),
            Tok::RParen => write!(f, ")"),
            Tok::LBrace => write!(f, "{{"),
            Tok::RBrace => write!(f, "}}"),
            Tok::LBracket => write!(f, "["),
            Tok::RBracket => write!(f, "]"),
            Tok::Colon => write!(f, ":"),
            Tok::Semicolon => write!(f, ";"),
            Tok::Comma => write!(f, ","),
            Tok::Ampersand => write!(f, "&"),
            Tok::And => write!(f, "&&"),
            Tok::Pipe => write!(f, "|"),
            Tok::Or => write!(f, "||"),
            Tok::Plus => write!(f, "+"),
            Tok::Minus => write!(f, "-"),
            Tok::Asterisk => write!(f, "*"),
            Tok::Slash => write!(f, "/"),
            Tok::Let => write!(f, "let"),
            Tok::Mut => write!(f, "mut"),
            Tok::ThinArrow => write!(f, "->"),
            Tok::ThickArrow => write!(f, "=>"),
            Tok::Fn => write!(f, "fn"),
            Tok::Not => write!(f, "!"),
            Tok::NotEquals => write!(f, "!="),
            Tok::True => write!(f, "true"),
            Tok::False => write!(f, "false"),
            Tok::Underscore => write!(f, "_"),
            Tok::Period => write!(f, "."),
            Tok::DoublePeriod => write!(f, ".."),
            Tok::If => write!(f, "if"),
            Tok::Else => write!(f, "else"),
            Tok::Percent => write!(f, "%"),
            Tok::PercentLBrace => write!(f, "%{{"),
            Tok::For => write!(f, "for"),
            Tok::In => write!(f, "in"),
            Tok::While => write!(f, "while"),
            Tok::Break => write!(f, "break"),
            Tok::Continue => write!(f, "continue"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexError {
    InvalidSymbol(String),
    InvalidNumber(String),
    UnterminatedString(String),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CharInfo {
    index: usize,
    row: usize,
    col: usize,
    ch: char,
}

pub struct Lexer {
    current_index: usize,
    chars: Vec<CharInfo>,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let mut row = 1;
        let mut col = 1;
        let mut chars = vec![];
        for (index, ch) in input.char_indices() {
            chars.push(CharInfo {
                index,
                row,
                col,
                ch,
            });
            if ch == '\n' {
                row += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        Lexer {
            current_index: 0,
            chars,
        }
    }

    pub fn peek_char(&mut self, index: usize) -> Option<CharInfo> {
        if self.current_index + index >= self.chars.len() {
            None
        } else {
            Some(self.chars[self.current_index + index])
        }
    }

    pub fn lex_char(&mut self) -> CharInfo {
        let char_info = self.chars[self.current_index];
        self.current_index += 1;
        char_info
    }

    fn lex_whitespace(&mut self) {
        while let Some(char_info) = self.peek_char(0) {
            if char_info.ch.is_whitespace() {
                self.lex_char();
            } else {
                break;
            }
        }
    }

    fn lex_identifier(&mut self) -> Span {
        let start_char_info = self.lex_char();
        let mut end_char_info = start_char_info;
        let ch = start_char_info.ch;
        assert!(ch.is_alphabetic() || ch == '_');
        while let Some(char_info) = self.peek_char(0) {
            let ch = char_info.ch;
            if ch.is_alphanumeric() || ch == '_' {
                end_char_info = self.lex_char();
            } else {
                break;
            }
        }
        let identifier: String = self.chars[start_char_info.index..=end_char_info.index]
            .iter()
            .map(|char_info| char_info.ch)
            .collect();
        let tok = match identifier.as_ref() {
            "type" => Tok::Type,
            "let" => Tok::Let,
            "mut" => Tok::Mut,
            "fn" => Tok::Fn,
            "true" => Tok::True,
            "false" => Tok::False,
            "_" => Tok::Underscore,
            "if" => Tok::If,
            "else" => Tok::Else,
            "for" => Tok::For,
            "in" => Tok::In,
            "while" => Tok::While,
            "break" => Tok::Break,
            "continue" => Tok::Continue,
            _ => Tok::Identifier(identifier),
        };
        (start_char_info.index, tok, end_char_info.index + 1)
    }

    fn lex_symbol(&mut self) -> Result {
        let char_info = self.lex_char();
        assert!(char_info.ch.is_ascii_punctuation());
        let tok = match char_info.ch {
            '(' => Some(Tok::LParen),
            ')' => Some(Tok::RParen),
            '{' => Some(Tok::LBrace),
            '}' => Some(Tok::RBrace),
            '[' => Some(Tok::LBracket),
            ']' => Some(Tok::RBracket),
            ':' => Some(Tok::Colon),
            ';' => Some(Tok::Semicolon),
            ',' => Some(Tok::Comma),
            '+' => Some(Tok::Plus),
            '.' => {
                if let Some(char_info) = self.peek_char(0) {
                    match char_info.ch {
                        '.' => {
                            self.lex_char();
                            Some(Tok::DoublePeriod)
                        }
                        _ => Some(Tok::Period),
                    }
                } else {
                    Some(Tok::Period)
                }
            }
            '-' => {
                if let Some(char_info) = self.peek_char(0) {
                    match char_info.ch {
                        '>' => {
                            self.lex_char();
                            Some(Tok::ThinArrow)
                        }
                        _ => Some(Tok::Minus),
                    }
                } else {
                    Some(Tok::Minus)
                }
            }
            '/' => Some(Tok::Slash),
            '*' => Some(Tok::Asterisk),
            '=' => {
                if let Some(char_info) = self.peek_char(0) {
                    match char_info.ch {
                        '=' => {
                            self.lex_char();
                            Some(Tok::DoubleEquals)
                        }
                        '>' => {
                            self.lex_char();
                            Some(Tok::ThickArrow)
                        }
                        _ => Some(Tok::Equals),
                    }
                } else {
                    Some(Tok::Equals)
                }
            }
            '&' => {
                if let Some(char_info) = self.peek_char(0) {
                    if char_info.ch == '&' {
                        self.lex_char();
                        Some(Tok::And)
                    } else {
                        Some(Tok::Ampersand)
                    }
                } else {
                    Some(Tok::Ampersand)
                }
            }
            '|' => {
                if let Some(char_info) = self.peek_char(0) {
                    if char_info.ch == '|' {
                        self.lex_char();
                        Some(Tok::Or)
                    } else {
                        Some(Tok::Pipe)
                    }
                } else {
                    Some(Tok::Pipe)
                }
            }
            '!' => {
                if let Some(char_info) = self.peek_char(0) {
                    if char_info.ch == '=' {
                        self.lex_char();
                        Some(Tok::NotEquals)
                    } else {
                        Some(Tok::Not)
                    }
                } else {
                    Some(Tok::Not)
                }
            }
            '%' => {
                if let Some(char_info) = self.peek_char(0) {
                    if char_info.ch == '{' {
                        self.lex_char();
                        Some(Tok::PercentLBrace)
                    } else {
                        Some(Tok::Percent)
                    }
                } else {
                    Some(Tok::Percent)
                }
            }
            _ => unreachable!(&format!("unhandled symbol: {}", char_info.ch)),
        };
        if let Some(tok) = tok {
            let len = tok.to_string().len();
            Ok((char_info.index, tok, char_info.index + len))
        } else {
            Err(LexError::InvalidSymbol(char_info.ch.to_string()))
        }
    }

    fn lex_number(&mut self) -> Result {
        let start_char_info = self.lex_char();
        let mut end_char_info = start_char_info;
        let ch = start_char_info.ch;
        assert!(ch.is_numeric());
        while let Some(char_info) = self.peek_char(0) {
            let ch = char_info.ch;
            if ch.is_numeric() {
                end_char_info = self.lex_char();
            } else {
                break;
            }
        }

        let mut tail = end_char_info;
        while let Some(char_info) = self.peek_char(0) {
            if char_info.ch.is_alphanumeric() {
                tail = self.lex_char();
            } else {
                break;
            }
        }

        let number_str: String = self.chars[start_char_info.index..=tail.index]
            .iter()
            .map(|char_info| char_info.ch)
            .collect();

        if tail != end_char_info {
            return Err(LexError::InvalidNumber(number_str));
        }

        let number: usize = number_str.parse().unwrap();
        Ok((
            start_char_info.index,
            Tok::Number(number),
            end_char_info.index + 1,
        ))
    }

    fn lex_string(&mut self) -> Result {
        let start_char_info = self.lex_char();
        let mut end_char_info = start_char_info;
        assert!(start_char_info.ch == '"');
        while let Some(char_info) = self.peek_char(0) {
            self.lex_char();
            if char_info.ch == '"' {
                end_char_info = char_info;
                break;
            }
        }

        if end_char_info == start_char_info {
            // Reached end without seeing closing quote
            let s = self.chars[start_char_info.index + 1..self.chars.len()]
                .iter()
                .map(|char_info| char_info.ch)
                .collect();
            return Err(LexError::UnterminatedString(s));
        }

        let s = self.chars[start_char_info.index + 1..end_char_info.index]
            .iter()
            .map(|char_info| char_info.ch)
            .collect();
        Ok((
            start_char_info.index,
            Tok::String(s),
            end_char_info.index + 1,
        ))
    }
}

impl Iterator for Lexer {
    type Item = Result;

    fn next(&mut self) -> Option<Self::Item> {
        self.lex_whitespace();
        if self.current_index >= self.chars.len() {
            return None;
        }
        if let Some(char_info) = self.peek_char(0) {
            if SYMBOL_START_CHARS.contains(&char_info.ch) {
                Some(self.lex_symbol())
            } else if char_info.ch.is_alphabetic() || char_info.ch == '_' {
                Some(Ok(self.lex_identifier()))
            } else if char_info.ch.is_numeric() {
                Some(self.lex_number())
            } else if char_info.ch == '"' {
                Some(self.lex_string())
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_identifier() {
        let mut lexer = Lexer::new("asdf");
        assert_eq!(
            lexer.lex_identifier(),
            (0, Tok::Identifier("asdf".to_owned()), 4)
        );
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_keyword() {
        let mut lexer = Lexer::new("type");
        assert_eq!(lexer.lex_identifier(), (0, Tok::Type, 4));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_symbol() {
        let mut lexer = Lexer::new("&&");
        assert_eq!(lexer.lex_symbol(), Ok((0, Tok::And, 2)));
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new("&=->");
        assert_eq!(lexer.next(), Some(Ok((0, Tok::Ampersand, 1))));
        assert_eq!(lexer.next(), Some(Ok((1, Tok::Equals, 2))));
        assert_eq!(lexer.next(), Some(Ok((2, Tok::ThinArrow, 4))));
    }

    #[test]
    fn test_lex_whitespace_and_identifier() {
        let mut lexer = Lexer::new("    asdf");
        lexer.lex_whitespace();
        assert_eq!(
            lexer.lex_identifier(),
            (4, Tok::Identifier("asdf".to_owned()), 8)
        );
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_number() {
        let mut lexer = Lexer::new("123");
        assert_eq!(lexer.lex_number(), Ok((0, Tok::Number(123), 3)));
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new("123a");
        assert_eq!(
            lexer.lex_number(),
            Err(LexError::InvalidNumber("123a".to_owned()))
        );
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lex_string() {
        let mut lexer = Lexer::new("\"asdf\"");
        assert_eq!(
            lexer.lex_string(),
            Ok((0, Tok::String("asdf".to_owned()), 6))
        );
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new("\"asdf");
        assert_eq!(
            lexer.lex_string(),
            Err(LexError::UnterminatedString("asdf".to_owned()))
        );
        assert_eq!(lexer.next(), None);
    }
}
