extern crate rand;

use rand::{Rand, Rng, StdRng};

pub const INITIAL_WINDOW_WIDTH: u32 = 800;
pub const INITIAL_WINDOW_HEIGHT: u32 = 600;

pub fn random_string(rng: &mut StdRng) -> String {
    rng.gen_ascii_chars().take(6).collect()
}

pub struct Platform {
    pub draw_poly: fn(f32, f32, usize, usize),
    pub draw_poly_with_matrix: fn([f32; 16], usize, usize),
    pub draw_poly_with_matrix_and_colours: fn([f32; 16], rgba, rgba, usize, usize),
    pub draw_textured_poly: fn(f32, f32, usize, TextureSpec, usize),
    pub draw_textured_poly_with_matrix: fn([f32; 16], usize, TextureSpec, usize),
    pub draw_text: fn(&str, (f32, f32), f32, f32, [f32; 4], usize),
    pub draw_layer: fn(usize, f32),
    pub set_verts: fn(Vec<Vec<f32>>),
}

pub struct State {
    pub rng: StdRng,
    pub cam_x: f32,
    pub cam_y: f32,
    pub zoom: f32,
    pub mouse_pos: (f32, f32),
    pub mouse_held: bool,
    pub window_wh: (f32, f32),
    pub ui_context: UIContext,
    pub x: f32,
    pub y: f32,
    pub screens: Vec<Screen>,
    pub portal_smell: u64,
    pub phase: Phase,
    pub screen_index: ScreenIndex,
}

#[derive(Debug)]
pub enum Phase {
    Move,
    PlaceFirstPortal((ScreenIndex, ScreenIndex), ScreenIndex),
    PlaceSecondPortal(ScreenIndex, PortalSpec, ScreenIndex),
    PlaceFunctionPortal(ScreenIndex, PortalFunction),
}

#[derive(Debug)]
pub struct PlainPortal {
    pub x: f32,
    pub y: f32,
    pub target: PortalTarget,
}

impl PlainPortal {
    pub fn from_spec(spec: PortalSpec, target: PortalTarget) -> Self {
        PlainPortal {
            x: spec.x,
            y: spec.y,
            target,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PortalSpec {
    pub x: f32,
    pub y: f32,
    pub screen_index: ScreenIndex,
}

#[derive(Debug, Clone)]
pub struct PortalTarget {
    pub screen_index: ScreenIndex,
    pub target: usize,
}

#[derive(Debug)]
pub struct FunctionPortal {
    pub x: f32,
    pub y: f32,
    pub target_screen_index: ScreenIndex,
    pub function: PortalFunction,
}

pub struct PortalFunction(pub Box<Fn(f32, f32) -> (f32, f32)>);

impl PortalFunction {
    pub fn new(f: Box<Fn(f32, f32) -> (f32, f32)>) -> Self {
        PortalFunction(f)
    }
}

use std::fmt;
impl fmt::Debug for PortalFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PortalFunction(closure)")
    }
}

pub type ScreenIndex = usize;

#[derive(Default, Debug)]
pub struct Screen {
    pub plain_portals: Vec<PlainPortal>,
    pub function_portals: Vec<FunctionPortal>,
    pub goals: Vec<Goal>,
    pub goal_nodes: Vec<GoalNode>,
}

#[derive(Debug)]
pub struct Goal {
    pub x: f32,
    pub y: f32,
}

impl Rand for Goal {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        Goal {
            x: rng.gen_range(-0.875, 0.875),
            y: rng.gen_range(-0.875, 0.875),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GoalNode {
    pub x: f32,
    pub y: f32,
}

impl Rand for GoalNode {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        GoalNode {
            x: rng.gen_range(-0.875, 0.875),
            y: rng.gen_range(-0.875, 0.875),
        }
    }
}

//Rgba and RGBA both seem worse than this
#[allow(non_camel_case_types)]
pub type rgba = (f32, f32, f32, f32);

#[derive(Debug)]
pub enum Event {
    Quit,
    KeyDown(Keycode),
    KeyUp(Keycode),
    MouseMove((i32, i32)),
    LeftMouseUp,
    LeftMouseDown,
    RightMouseUp,
    RightMouseDown,
    WindowSize((i32, i32)),
}

pub type UiId = i32;

#[derive(Debug)]
pub struct UIContext {
    pub hot: UiId,
    pub active: UiId,
    pub next_hot: UiId,
}

impl UIContext {
    pub fn new() -> Self {
        UIContext {
            hot: 0,
            active: 0,
            next_hot: 0,
        }
    }

    pub fn set_not_active(&mut self) {
        self.active = 0;
    }
    pub fn set_active(&mut self, id: UiId) {
        self.active = id;
    }
    pub fn set_next_hot(&mut self, id: UiId) {
        self.next_hot = id;
    }
    pub fn set_not_hot(&mut self) {
        self.hot = 0;
    }
    pub fn frame_init(&mut self) {
        if self.active == 0 {
            self.hot = self.next_hot;
        }
        self.next_hot = 0;
    }
}


pub type TextureSpec = (f32, f32, f32, f32, i32, f32, f32, f32, f32);

pub struct ProjectionSpec {
    pub top: f32,
    pub bottom: f32,
    pub left: f32,
    pub right: f32,
    pub near: f32,
    pub far: f32,
    pub projection: Projection,
}

pub enum Projection {
    Perspective,
    Orthographic,
}
use Projection::*;

// see http://www.scratchapixel.com perspecitve and orthographic projection sections
pub fn get_projection(spec: &ProjectionSpec) -> [f32; 16] {
    match spec.projection {
        Perspective => {
            [
                (2.0 * spec.near) / (spec.right - spec.left),
                0.0,
                (spec.right + spec.left) / (spec.right - spec.left),
                0.0,
                //
                0.0,
                (2.0 * spec.near) / (spec.top - spec.bottom),
                (spec.top + spec.bottom) / (spec.top - spec.bottom),
                0.0,
                //
                0.0,
                0.0,
                -(spec.far + spec.near) / (spec.far - spec.near),
                -(2.0 * spec.far * spec.near) / (spec.far - spec.near),
                //
                0.0,
                0.0,
                -1.0,
                0.0,
            ]
        }
        Orthographic => {
            [
                2.0 / (spec.right - spec.left),
                0.0,
                0.0,
                -(spec.right + spec.left) / (spec.right - spec.left),
                //
                0.0,
                2.0 / (spec.top - spec.bottom),
                0.0,
                -(spec.top + spec.bottom) / (spec.top - spec.bottom),
                //
                0.0,
                0.0,
                -2.0 / (spec.far - spec.near),
                -(spec.far + spec.near) / (spec.far - spec.near),
                //
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        }
    }
}

pub const IDENTITY_MAT4X4: [f32; 16] = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
];

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
mod mat4x4_tests {
    use ::*;

    impl Rand for ProjectionSpec {
        fn rand<R: Rng>(rng: &mut R) -> Self {
            ProjectionSpec {
                top: rng.gen(),
                bottom: rng.gen(),
                left: rng.gen(),
                right: rng.gen(),
                near: rng.gen(),
                far: rng.gen(),
                projection: rng.gen(),
            }
        }
    }

    impl quickcheck::Arbitrary for ProjectionSpec {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> ProjectionSpec {
            g.gen()
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct Mat4x4 {
        m: [f32; 16],
    }

    impl Rand for Mat4x4 {
        fn rand<R: Rng>(rng: &mut R) -> Self {
            let m = [
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
            ];

            Mat4x4 { m }
        }
    }

    impl quickcheck::Arbitrary for Mat4x4 {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Mat4x4 {
            g.gen()
        }
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    quickcheck! {
        fn projection_inversion(mat4x4: Mat4x4, spec_: ProjectionSpec) -> bool {
            let m = mat4x4.m;

            let mut spec = spec_.clone();
            spec.projection = Orthographic;
            let p = get_projection(&spec);
            let p_inv = get_projection(&spec.inverse());

            let actual = mat4x4_mul(&p_inv, &mat4x4_mul(&p,&m));

            // m ~= actual
            let mut error:f32 = 0.0;
            for i in 0..actual.len() {
                let current_error = (actual[i] - m[i]).abs();
                error = if error > current_error {
                    error
                } else {
                    current_error
                };
            }

            error <= 0.0001
        }

        fn projection_camera_inversion(
            mat4x4: Mat4x4,
            spec_: ProjectionSpec,
            cam_x: f32,
            cam_y:f32
        )
            -> bool
        {
            let m = mat4x4.m;

            let mut spec = spec_.clone();
            spec.projection = Orthographic;
            let p = get_projection(&spec);
            let p_inv = get_projection(&spec.inverse());

            let camera = [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                cam_x,
                cam_y,
                0.0,
                1.0,
            ];

            let camera_inv = [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                -cam_x,
                -cam_y,
                0.0,
                1.0,
            ];

            let view = mat4x4_mul(&camera, &p);
            let view_inv = mat4x4_mul(&p_inv, &camera_inv);

            let actual = mat4x4_mul(&view_inv, &mat4x4_mul(&view,&m));

            // m ~= actual
            let mut error:f32 = 0.0;
            for i in 0..actual.len() {
                let current_error = (actual[i] - m[i]).abs();
                error = if error > current_error {
                    error
                } else {
                    current_error
                };
            }
            println!("error {}", error);
            error <= 0.25
        }
    }

}

pub fn mat4x4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    [
        a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12],
        a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13],
        a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14],
        a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15],
        a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12],
        a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13],
        a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14],
        a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15],
        a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12],
        a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13],
        a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14],
        a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15],
        a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12],
        a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13],
        a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14],
        a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15],
    ]
}

pub fn mat4x4_vector_mul(
    m: &[f32; 16],
    v_0: f32,
    v_1: f32,
    v_2: f32,
    v_3: f32,
) -> (f32, f32, f32, f32) {
    (
        m[0] * v_0 + m[4] * v_1 + m[8] * v_2 + m[12] * v_3,
        m[1] * v_0 + m[5] * v_1 + m[9] * v_2 + m[13] * v_3,
        m[2] * v_0 + m[6] * v_1 + m[10] * v_2 + m[14] * v_3,
        m[3] * v_0 + m[7] * v_1 + m[11] * v_2 + m[15] * v_3,
    )
}
pub fn mat4x4_vector_mul_divide(
    m: &[f32; 16],
    v_0: f32,
    v_1: f32,
    v_2: f32,
    v_3: f32,
) -> (f32, f32, f32, f32) {
    let (x, y, z, w) = mat4x4_vector_mul(m, v_0, v_1, v_2, v_3);
    (x / w, y / w, z / w, 1.0)
}

// Precise method, which guarantees v = v1 when t = 1.
pub fn lerp(v0: f32, v1: f32, t: f32) -> f32 {
    (1.0 - t) * v0 + t * v1
}

//combined from https://github.com/AngryLawyer/rust-sdl2/blob/master/sdl2-sys/src/keycode.rs
// and https://github.com/AngryLawyer/rust-sdl2/blob/master/src/sdl2/keyboard/keycode.rs
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(i32)]
pub enum Keycode {
    Backspace = 8 as i32,
    Tab = 9 as i32,
    Return = 13 as i32,
    Escape = 27 as i32,
    Space = 32 as i32,
    Exclaim = 33 as i32,
    Quotedbl = 34 as i32,
    Hash = 35 as i32,
    Dollar = 36 as i32,
    Percent = 37 as i32,
    Ampersand = 38 as i32,
    Quote = 39 as i32,
    LeftParen = 40 as i32,
    RightParen = 41 as i32,
    Asterisk = 42 as i32,
    Plus = 43 as i32,
    Comma = 44 as i32,
    Minus = 45 as i32,
    Period = 46 as i32,
    Slash = 47 as i32,
    Num0 = 48 as i32,
    Num1 = 49 as i32,
    Num2 = 50 as i32,
    Num3 = 51 as i32,
    Num4 = 52 as i32,
    Num5 = 53 as i32,
    Num6 = 54 as i32,
    Num7 = 55 as i32,
    Num8 = 56 as i32,
    Num9 = 57 as i32,
    Colon = 58 as i32,
    Semicolon = 59 as i32,
    Less = 60 as i32,
    Equals = 61 as i32,
    Greater = 62 as i32,
    Question = 63 as i32,
    At = 64 as i32,
    LeftBracket = 91 as i32,
    Backslash = 92 as i32,
    RightBracket = 93 as i32,
    Caret = 94 as i32,
    Underscore = 95 as i32,
    Backquote = 96 as i32,
    A = 97 as i32,
    B = 98 as i32,
    C = 99 as i32,
    D = 100 as i32,
    E = 101 as i32,
    F = 102 as i32,
    G = 103 as i32,
    H = 104 as i32,
    I = 105 as i32,
    J = 106 as i32,
    K = 107 as i32,
    L = 108 as i32,
    M = 109 as i32,
    N = 110 as i32,
    O = 111 as i32,
    P = 112 as i32,
    Q = 113 as i32,
    R = 114 as i32,
    S = 115 as i32,
    T = 116 as i32,
    U = 117 as i32,
    V = 118 as i32,
    W = 119 as i32,
    X = 120 as i32,
    Y = 121 as i32,
    Z = 122 as i32,
    Delete = 127 as i32,
    CapsLock = 1073741881 as i32,
    F1 = 1073741882 as i32,
    F2 = 1073741883 as i32,
    F3 = 1073741884 as i32,
    F4 = 1073741885 as i32,
    F5 = 1073741886 as i32,
    F6 = 1073741887 as i32,
    F7 = 1073741888 as i32,
    F8 = 1073741889 as i32,
    F9 = 1073741890 as i32,
    F10 = 1073741891 as i32,
    F11 = 1073741892 as i32,
    F12 = 1073741893 as i32,
    PrintScreen = 1073741894 as i32,
    ScrollLock = 1073741895 as i32,
    Pause = 1073741896 as i32,
    Insert = 1073741897 as i32,
    Home = 1073741898 as i32,
    PageUp = 1073741899 as i32,
    End = 1073741901 as i32,
    PageDown = 1073741902 as i32,
    Right = 1073741903 as i32,
    Left = 1073741904 as i32,
    Down = 1073741905 as i32,
    Up = 1073741906 as i32,
    NumLockClear = 1073741907 as i32,
    KpDivide = 1073741908 as i32,
    KpMultiply = 1073741909 as i32,
    KpMinus = 1073741910 as i32,
    KpPlus = 1073741911 as i32,
    KpEnter = 1073741912 as i32,
    Kp1 = 1073741913 as i32,
    Kp2 = 1073741914 as i32,
    Kp3 = 1073741915 as i32,
    Kp4 = 1073741916 as i32,
    Kp5 = 1073741917 as i32,
    Kp6 = 1073741918 as i32,
    Kp7 = 1073741919 as i32,
    Kp8 = 1073741920 as i32,
    Kp9 = 1073741921 as i32,
    Kp0 = 1073741922 as i32,
    KpPeriod = 1073741923 as i32,
    Application = 1073741925 as i32,
    Power = 1073741926 as i32,
    KpEquals = 1073741927 as i32,
    F13 = 1073741928 as i32,
    F14 = 1073741929 as i32,
    F15 = 1073741930 as i32,
    F16 = 1073741931 as i32,
    F17 = 1073741932 as i32,
    F18 = 1073741933 as i32,
    F19 = 1073741934 as i32,
    F20 = 1073741935 as i32,
    F21 = 1073741936 as i32,
    F22 = 1073741937 as i32,
    F23 = 1073741938 as i32,
    F24 = 1073741939 as i32,
    Execute = 1073741940 as i32,
    Help = 1073741941 as i32,
    Menu = 1073741942 as i32,
    Select = 1073741943 as i32,
    Stop = 1073741944 as i32,
    Again = 1073741945 as i32,
    Undo = 1073741946 as i32,
    Cut = 1073741947 as i32,
    Copy = 1073741948 as i32,
    Paste = 1073741949 as i32,
    Find = 1073741950 as i32,
    Mute = 1073741951 as i32,
    VolumeUp = 1073741952 as i32,
    VolumeDown = 1073741953 as i32,
    KpComma = 1073741957 as i32,
    KpEqualsAS400 = 1073741958 as i32,
    AltErase = 1073741977 as i32,
    Sysreq = 1073741978 as i32,
    Cancel = 1073741979 as i32,
    Clear = 1073741980 as i32,
    Prior = 1073741981 as i32,
    Return2 = 1073741982 as i32,
    Separator = 1073741983 as i32,
    Out = 1073741984 as i32,
    Oper = 1073741985 as i32,
    ClearAgain = 1073741986 as i32,
    CrSel = 1073741987 as i32,
    ExSel = 1073741988 as i32,
    Kp00 = 1073742000 as i32,
    Kp000 = 1073742001 as i32,
    ThousandsSeparator = 1073742002 as i32,
    DecimalSeparator = 1073742003 as i32,
    CurrencyUnit = 1073742004 as i32,
    CurrencySubUnit = 1073742005 as i32,
    KpLeftParen = 1073742006 as i32,
    KpRightParen = 1073742007 as i32,
    KpLeftBrace = 1073742008 as i32,
    KpRightBrace = 1073742009 as i32,
    KpTab = 1073742010 as i32,
    KpBackspace = 1073742011 as i32,
    KpA = 1073742012 as i32,
    KpB = 1073742013 as i32,
    KpC = 1073742014 as i32,
    KpD = 1073742015 as i32,
    KpE = 1073742016 as i32,
    KpF = 1073742017 as i32,
    KpXor = 1073742018 as i32,
    KpPower = 1073742019 as i32,
    KpPercent = 1073742020 as i32,
    KpLess = 1073742021 as i32,
    KpGreater = 1073742022 as i32,
    KpAmpersand = 1073742023 as i32,
    KpDblAmpersand = 1073742024 as i32,
    KpVerticalBar = 1073742025 as i32,
    KpDblVerticalBar = 1073742026 as i32,
    KpColon = 1073742027 as i32,
    KpHash = 1073742028 as i32,
    KpSpace = 1073742029 as i32,
    KpAt = 1073742030 as i32,
    KpExclam = 1073742031 as i32,
    KpMemStore = 1073742032 as i32,
    KpMemRecall = 1073742033 as i32,
    KpMemClear = 1073742034 as i32,
    KpMemAdd = 1073742035 as i32,
    KpMemSubtract = 1073742036 as i32,
    KpMemMultiply = 1073742037 as i32,
    KpMemDivide = 1073742038 as i32,
    KpPlusMinus = 1073742039 as i32,
    KpClear = 1073742040 as i32,
    KpClearEntry = 1073742041 as i32,
    KpBinary = 1073742042 as i32,
    KpOctal = 1073742043 as i32,
    KpDecimal = 1073742044 as i32,
    KpHexadecimal = 1073742045 as i32,
    LCtrl = 1073742048 as i32,
    LShift = 1073742049 as i32,
    LAlt = 1073742050 as i32,
    LGui = 1073742051 as i32,
    RCtrl = 1073742052 as i32,
    RShift = 1073742053 as i32,
    RAlt = 1073742054 as i32,
    RGui = 1073742055 as i32,
    Mode = 1073742081 as i32,
    AudioNext = 1073742082 as i32,
    AudioPrev = 1073742083 as i32,
    AudioStop = 1073742084 as i32,
    AudioPlay = 1073742085 as i32,
    AudioMute = 1073742086 as i32,
    MediaSelect = 1073742087 as i32,
    Www = 1073742088 as i32,
    Mail = 1073742089 as i32,
    Calculator = 1073742090 as i32,
    Computer = 1073742091 as i32,
    AcSearch = 1073742092 as i32,
    AcHome = 1073742093 as i32,
    AcBack = 1073742094 as i32,
    AcForward = 1073742095 as i32,
    AcStop = 1073742096 as i32,
    AcRefresh = 1073742097 as i32,
    AcBookmarks = 1073742098 as i32,
    BrightnessDown = 1073742099 as i32,
    BrightnessUp = 1073742100 as i32,
    DisplaySwitch = 1073742101 as i32,
    KbdIllumToggle = 1073742102 as i32,
    KbdIllumDown = 1073742103 as i32,
    KbdIllumUp = 1073742104 as i32,
    Eject = 1073742105 as i32,
    Sleep = 1073742106 as i32,
}
