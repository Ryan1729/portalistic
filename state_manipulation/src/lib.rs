extern crate common;
extern crate rand;

use common::*;
use common::Projection::*;
use common::Phase::*;

use rand::{Rng, SeedableRng, StdRng};

#[cfg(debug_assertions)]
#[no_mangle]
pub fn new_state() -> State {
    println!("debug on");

    let seed: &[_] = &[42];
    let rng: StdRng = SeedableRng::from_seed(seed);

    make_state(rng)
}
#[cfg(not(debug_assertions))]
#[no_mangle]
pub fn new_state() -> State {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|dur| dur.as_secs())
        .unwrap_or(42);

    println!("{}", timestamp);
    let seed: &[_] = &[timestamp as usize];
    let rng: StdRng = SeedableRng::from_seed(seed);

    make_state(rng)
}

fn make_state(mut rng: StdRng) -> State {
    let mut portals = Vec::new();

    add_random_portal_pair(&mut rng, &mut portals);

    let mut state = State {
        rng,
        cam_x: 0.0,
        cam_y: 0.0,
        zoom: 1.0,
        mouse_pos: (400.0, 300.0),
        mouse_held: false,
        window_wh: (INITIAL_WINDOW_WIDTH as _, INITIAL_WINDOW_HEIGHT as _),
        ui_context: UIContext::new(),
        x: 0.0,
        y: 0.0,
        portals,
        portal_smell: 0,
        goals: vec![rng.gen()],
        phase: Move,
    };

    state
}

fn add_random_portal_pair(rng: &mut StdRng, portals: &mut Vec<Portal>) {
    let x1 = rng.gen_range(-0.875, 0.875);
    let y1 = rng.gen_range(-0.875, 0.875);
    let x2 = rng.gen_range(-0.875, 0.875);
    let y2 = rng.gen_range(-0.875, 0.875);

    add_first_portal(portals, x1, y1);
    add_second_portal(portals, x2, y2);
}

fn add_first_portal(portals: &mut Vec<Portal>, x: f32, y: f32) {
    let target = portals.len() + 1;

    portals.push(Portal { x, y, target });
}
fn add_second_portal(portals: &mut Vec<Portal>, x: f32, y: f32) {
    let target = portals.len() - 1;

    portals.push(Portal { x, y, target });
}

const TRANSLATION_SCALE: f32 = 0.0625;

#[no_mangle]
//returns true if quit requested
//lag is expected to be in nanoseconds
pub fn update_and_render(
    p: &Platform,
    state: &mut State,
    events: &mut Vec<Event>,
    lag: &mut u32,
) -> bool {
    let mut mouse_pressed = false;
    let mut mouse_released = false;

    for event in events {
        if cfg!(debug_assertions) {
            match *event {
                Event::MouseMove(_) => {}
                _ => println!("{:?}", *event),
            }
        }

        match *event {
            Event::Quit | Event::KeyDown(Keycode::Escape) | Event::KeyDown(Keycode::F10) => {
                return true;
            }
            Event::KeyDown(Keycode::Num9) => {
                add_random_portal_pair(&mut state.rng, &mut state.portals);
            }
            Event::KeyDown(Keycode::P) => {
                state.phase = PlaceFirstPortal;
            }
            Event::KeyDown(Keycode::L) => {
                state.cam_y += state.zoom * TRANSLATION_SCALE;
            }
            Event::KeyDown(Keycode::K) => {
                state.cam_y -= state.zoom * TRANSLATION_SCALE;
            }
            Event::KeyDown(Keycode::Semicolon) => {
                state.cam_x += state.zoom * TRANSLATION_SCALE;
            }
            Event::KeyDown(Keycode::J) => {
                state.cam_x -= state.zoom * TRANSLATION_SCALE;
            }
            Event::KeyDown(Keycode::Up) => {
                state.cam_y += state.zoom * TRANSLATION_SCALE;
            }
            Event::KeyDown(Keycode::Down) => {
                state.cam_y -= state.zoom * TRANSLATION_SCALE;
            }
            Event::KeyDown(Keycode::Right) => {
                state.cam_x += state.zoom * TRANSLATION_SCALE;
            }
            Event::KeyDown(Keycode::Left) => {
                state.cam_x -= state.zoom * TRANSLATION_SCALE;
            }
            Event::KeyDown(Keycode::Num0) => {
                state.cam_x = 0.0;
                state.cam_y = 0.0;
                state.zoom = 1.0;
            }
            Event::KeyDown(Keycode::R) => if cfg!(debug_assertions) {
                *state = new_state();
            },
            Event::KeyDown(Keycode::S) => {
                state.zoom *= 1.25;
            }
            Event::KeyDown(Keycode::W) => {
                state.zoom /= 1.25;
                if state.zoom == 0.0 {
                    state.zoom = std::f32::MIN_POSITIVE / TRANSLATION_SCALE;
                }
            }
            Event::MouseMove((x, y)) => {
                state.mouse_pos = (x as f32, y as f32);
            }
            Event::LeftMouseDown => {
                mouse_pressed = true;
            }
            Event::LeftMouseUp => {
                mouse_released = true;
            }
            Event::WindowSize((w, h)) => {
                state.window_wh = (w as f32, h as f32);
                if cfg!(debug_assertions) {
                    println!("aspect ratio: {}", state.window_wh.0 / state.window_wh.1);
                }
            }
            _ => {}
        }
    }

    if mouse_released != mouse_pressed {
        if mouse_released {
            state.mouse_held = false;
        } else {
            state.mouse_held = true;
        }
    }

    let mouse_button_state = ButtonState {
        pressed: mouse_pressed,
        released: mouse_released,
        held: state.mouse_held,
    };

    //map [0,1] to [-1,1]
    fn center(x: f32) -> f32 {
        x * 2.0 - 1.0
    }

    let mouse_x = center((state.mouse_pos.0) / state.window_wh.0);
    let mouse_y = -center(((state.mouse_pos.1) / state.window_wh.1));

    state.ui_context.frame_init();

    let aspect_ratio = 800.0 / 600.0;
    let near = 0.5;
    let far = 1024.0;

    let scale = state.zoom * near;
    let top = scale;
    let bottom = -top;
    let right = aspect_ratio * scale;
    let left = -right;

    let projection = get_projection(&ProjectionSpec {
        top,
        bottom,
        left,
        right,
        near,
        far,
        projection: Perspective,
        // projection: Orthographic,
    });

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
        state.cam_x,
        state.cam_y,
        0.0,
        1.0,
    ];

    let view = mat4x4_mul(&camera, &projection);

    let background_button_outcome = button_logic(
        &mut state.ui_context,
        Button {
            id: 1,
            pointer_inside: true,
            state: mouse_button_state,
        },
    );

    labeled_button(
        p,
        &mut state.ui_context,
        "Menu",
        (-0.75, 0.875),
        12,
        (mouse_x, mouse_y),
        mouse_button_state,
    );



    // (p.draw_text)(
    //     &format!("lag ms: {:.5}", *lag / 1_000),
    //     (0.0, 0.875),
    //     1.0,
    //     36.0,
    //     [0.0, 1.0, 1.0, 0.5],
    //     0,
    // );



    match state.phase {
        PlaceFirstPortal => if background_button_outcome.clicked {
            add_first_portal(&mut state.portals, mouse_x, mouse_y);

            state.phase = PlaceSecondPortal;
        },
        PlaceSecondPortal => if background_button_outcome.clicked {
            add_second_portal(&mut state.portals, mouse_x, mouse_y);

            state.phase = Move;
        },
        _ => {}
    }


    const PORTAL_SMELL_NS_PER: u64 = 3_000_000_000;
    const NS_PER_UPDATE: u32 = 1_000_000;

    while *lag >= NS_PER_UPDATE {
        match state.phase {
            Move => {
                let moving = match background_button_outcome.draw_state {
                    Pressed => true,
                    _ => false,
                };

                if moving {
                    let player_speed: f32 = 1.0 / 2f32.powi(12);

                    let dx = mouse_x - state.x;
                    let dy = mouse_y - state.y;

                    let angle = f32::atan2(dy, dx);

                    state.x += angle.cos() * player_speed;
                    state.y += angle.sin() * player_speed;
                }

                if state.portal_smell == 0 {
                    if let Some((x, y)) =
                        overlapping_portal_target_coords(&state.portals, state.x, state.y)
                    {
                        state.x = x;
                        state.y = y;

                        state.portal_smell += PORTAL_SMELL_NS_PER;
                    }
                }

                if let Some(i) = overlapping_goal_index(&state.goals, state.x, state.y) {
                    state.goals.swap_remove(i);

                    state.goals.push(state.rng.gen());
                }

                state.portal_smell = state.portal_smell.saturating_sub(NS_PER_UPDATE as _);
            }
            _ => {}
        }

        *lag = lag.saturating_sub(NS_PER_UPDATE);
    }

    (p.draw_text)(
        &format!("({:.5}, {:.5})", state.x, state.y),
        (0.0, -0.875),
        1.0,
        36.0,
        [0.0, 1.0, 1.0, 0.5],
        0,
    );

    let draw_portal = |x: f32, y: f32| {
        (p.draw_poly_with_matrix)(
            mat4x4_mul(&view, &scale_translation(1.0 / 16.0, x, y)),
            4,
            0,
        );
    };

    for portal in state.portals.iter() {
        draw_portal(portal.x, portal.y);
    }
    for goal in state.goals.iter() {
        (p.draw_poly_with_matrix_and_colours)(
            mat4x4_mul(&view, &scale_translation(1.0 / 16.0, goal.x, goal.y)),
            (192.0 / 255.0, 48.0 / 255.0, 48.0 / 255.0, 0.375),
            (192.0 / 255.0, 192.0 / 255.0, 48.0 / 255.0, 1.0),
            3,
            0,
        );
    }

    let mut smell_fraction = state.portal_smell as f32 / PORTAL_SMELL_NS_PER as f32;

    smell_fraction = if smell_fraction > 1.0 {
        1.0
    } else if smell_fraction < 0.0 {
        0.0
    } else {
        smell_fraction
    };

    (p.draw_poly_with_matrix_and_colours)(
        mat4x4_mul(&view, &scale_translation(1.0 / 16.0, state.x, state.y)),
        (
            smell_fraction,
            smell_fraction,
            if smell_fraction > 63.0 / 255.0 {
                smell_fraction
            } else {
                63.0 / 255.0
            },
            1.0,
        ),
        (192.0 / 255.0, 192.0 / 255.0, 48.0 / 255.0, 1.0),
        1,
        0,
    );

    match state.phase {
        PlaceFirstPortal | PlaceSecondPortal => {
            draw_portal(mouse_x, mouse_y);
        }
        _ => {}
    }

    false
}

fn overlapping_goal_index(goals: &Vec<Goal>, x: f32, y: f32) -> Option<usize> {
    //We can just cap how many goals the player can have at once
    //so O(N) will probably be fine.

    let mut result = None;

    const MINIMUM_DISTANCE_SQ: f32 = 0.0005;

    let mut smallest_so_far = std::f32::INFINITY;

    for (i, goal) in goals.iter().enumerate() {
        let distance_sq = get_euclidean_sq((x, y), (goal.x, goal.y));

        if distance_sq < MINIMUM_DISTANCE_SQ && MINIMUM_DISTANCE_SQ < smallest_so_far {
            result = Some(i);

            smallest_so_far = distance_sq;
        }
    }

    result
}

fn overlapping_portal_target_coords(portals: &Vec<Portal>, x: f32, y: f32) -> Option<(f32, f32)> {
    //curently I'm not expecting more than  16-ish portals on a screen,
    //and eventually I expect screens to be separated out so O(N) will
    //probably be fine.

    let mut result = None;

    const MINIMUM_DISTANCE_SQ: f32 = 0.0005;

    let mut smallest_so_far = std::f32::INFINITY;

    for portal in portals.iter() {
        let distance_sq = get_euclidean_sq((x, y), (portal.x, portal.y));

        if distance_sq < MINIMUM_DISTANCE_SQ && MINIMUM_DISTANCE_SQ < smallest_so_far {
            if let Some(target_portal) = portals.get(portal.target) {
                result = Some((target_portal.x, target_portal.y));
            }

            smallest_so_far = distance_sq;
        }
    }

    result
}

fn get_euclidean_sq((x1, y1): (f32, f32), (x2, y2): (f32, f32)) -> f32 {
    (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
}

fn labeled_button(
    p: &Platform,
    context: &mut UIContext,
    label: &str,
    (x, y): (f32, f32),
    id: UiId,
    (mouse_x, mouse_y): (f32, f32),
    state: ButtonState,
) -> bool {
    let camera = scale_translation(0.0625, x, y);

    let inverse_camera = inverse_scale_translation(0.0625, x, y);

    let (box_mouse_x, box_mouse_y, _, _) =
        mat4x4_vector_mul(&inverse_camera, mouse_x, mouse_y, 0.0, 1.0);

    let pointer_inside = box_mouse_x.abs() <= RECT_W_H_RATIO && box_mouse_y.abs() <= 1.0;

    let button_outcome = button_logic(
        context,
        Button {
            id,
            pointer_inside,
            state,
        },
    );

    let (fill, outline) = match button_outcome.draw_state {
        Pressed => (
            (32.0 / 255.0, 32.0 / 255.0, 63.0 / 255.0, 1.0),
            (192.0 / 255.0, 192.0 / 255.0, 48.0 / 255.0, 1.0),
        ),
        Hover => (
            (63.0 / 255.0, 63.0 / 255.0, 128.0 / 255.0, 1.0),
            (192.0 / 255.0, 192.0 / 255.0, 48.0 / 255.0, 1.0),
        ),
        Inactive => (
            (63.0 / 255.0, 63.0 / 255.0, 128.0 / 255.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
        ),
    };

    (p.draw_poly_with_matrix_and_colours)(camera, fill, outline, 6, 0);


    let font_scale = if label.len() > 8 { 18.0 } else { 24.0 };

    (p.draw_text)(label, (x, y), 1.0, font_scale, [1.0; 4], 0);

    button_outcome.clicked
}

#[derive(Copy, Clone, Debug)]
struct Button {
    id: UiId,
    pointer_inside: bool,
    state: ButtonState,
}

#[derive(Copy, Clone, Debug)]
struct ButtonState {
    pressed: bool,
    released: bool,
    held: bool,
}

#[derive(Copy, Clone, Debug)]
struct ButtonOutcome {
    clicked: bool,
    draw_state: DrawState,
}

impl Default for ButtonOutcome {
    fn default() -> Self {
        ButtonOutcome {
            clicked: false,
            draw_state: Inactive,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum DrawState {
    Pressed,
    Hover,
    Inactive,
}
use DrawState::*;

///This function handles the logic for a given button and returns wheter it was clicked
///and the state of the button so it can be drawn properly elsestate of the button so
///it can be drawn properly elsewhere
fn button_logic(context: &mut UIContext, button: Button) -> ButtonOutcome {
    /// In order for this to work properly `context.frame_init();`
    /// must be called at the start of each frame, before this function is called
    let mut clicked = false;

    let inside = button.pointer_inside;

    let id = button.id;

    if context.active == id {
        if button.state.released {
            clicked = context.hot == id && inside;

            context.set_not_active();
        }
    } else if context.hot == id {
        if button.state.pressed {
            context.set_active(id);
        }
    }

    if inside {
        context.set_next_hot(id);
    }

    let draw_state = if context.active == id && (button.state.held || button.state.pressed) {
        Pressed
    } else if context.hot == id {
        Hover
    } else {
        Inactive
    };

    ButtonOutcome {
        clicked,
        draw_state,
    }
}

fn scale_translation(scale: f32, x_offest: f32, y_offset: f32) -> [f32; 16] {
    [
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        x_offest,
        y_offset,
        0.0,
        1.0,
    ]
}

fn inverse_scale_translation(scale: f32, x_offest: f32, y_offset: f32) -> [f32; 16] {
    scale_translation(1.0 / scale, -x_offest / scale, -y_offset / scale)
}

//These are the verticies of the polygons which can be drawn.
//The index refers to the index of the inner vector within the outer vecton.
#[cfg_attr(rustfmt, rustfmt_skip)]
#[no_mangle]
pub fn get_vert_vecs() -> Vec<Vec<f32>> {
    vec![
        // star heptagon
        vec![
            -0.012640, 0.255336,
            0.152259, 0.386185,
            0.223982, 0.275978,
            0.191749, 0.169082,
            0.396864, 0.121742,
            0.355419, -0.003047,
            0.251747, -0.044495,
            0.342622, -0.234376,
            0.219218, -0.279777,
            0.122174, -0.224565,
            0.030379, -0.414003,
            -0.082058, -0.345830,
            -0.099398, -0.235534,
            -0.304740, -0.281878,
            -0.321543, -0.151465,
            -0.246122, -0.069141,
            -0.410383, 0.062507,
            -0.318899, 0.156955,
            -0.207511, 0.149317,
            -0.207000, 0.359823,
            -0.076118, 0.347186,
            -0.012640, 0.255336,
        ],
        // heptagon
        vec![
            0.555765, -0.002168,
            0.344819, -0.435866,
            -0.125783, -0.541348,
            -0.501668, -0.239184,
            -0.499786, 0.243091,
            -0.121556, 0.542313,
            0.348209, 0.433163,
            0.555765, -0.002168,
        ],
        // star hexagon
        vec![
            0.267355, 0.153145,
            0.158858, 0.062321,
            0.357493, -0.060252,
            0.266305, -0.154964,
            0.133401, -0.106415,
            0.126567, -0.339724,
            -0.001050, -0.308109,
            -0.025457, -0.168736,
            -0.230926, -0.279472,
            -0.267355, -0.153145,
            -0.158858, -0.062321,
            -0.357493, 0.060252,
            -0.266305, 0.154964,
            -0.133401, 0.106415,
            -0.126567, 0.339724,
            0.001050, 0.308109,
            0.025457, 0.168736,
            0.230926, 0.279472,
            0.267355, 0.153145,
        ],
        //hexagon
        vec![
        0.002000, -0.439500,
        -0.379618, -0.221482,
        -0.381618, 0.218018,
        -0.002000, 0.439500,
        0.379618, 0.221482,
        0.381618, -0.218018,
        0.002000, -0.439500,
        ],
        //invert 7 point star
        vec![
        -1.037129, 0.000000,
        -0.487625, 0.071884,
        -0.036111, 0.158214,
        0.934421, 0.449993,
        0.470524, 0.146807,
        0.101182, -0.126878,
        -0.646639, -0.810860,
        -0.360230, -0.336421,
        -0.146212, 0.070412,
        0.230783, 1.011126,
        0.178589, 0.459403,
        0.162283, 0.000000,
        0.230783, -1.011126,
        0.038425, -0.491395,
        -0.146212, -0.070412,
        -0.646639, 0.810860,
        -0.247828, 0.426059,
        0.101182, 0.126878,
        0.934421, -0.449993,
        0.408145, -0.276338,
        -0.036111, -0.158214,
        -1.037129, -0.000000,
        ],
        //invert 6 point star
        vec![
        -1.037129, 0.000000,
        -0.583093, -0.055358,
        -0.204743, -0.117901,
        0.517890, -0.299004,
        0.243039, -0.029458,
        -0.000266, 0.236263,
        -0.518564, 0.898180,
        -0.339488, 0.477294,
        -0.204477, 0.118362,
        0.000000, -0.598008,
        0.096008, -0.225207,
        0.204477, 0.118362,
        0.518564, 0.898180,
        0.243605, 0.532652,
        0.000266, 0.236263,
        -0.517890, -0.299004,
        -0.147031, -0.195748,
        0.204743, -0.117901,
        1.037129, 0.000000,
        0.583093, 0.055358,
        0.204743, 0.117901,
        -0.517890, 0.299004,
        -0.243039, 0.029458,
        0.000266, -0.236263,
        0.518564, -0.898180,
        0.339488, -0.477294,
        0.204477, -0.118362,
        -0.000000, 0.598008,
        -0.096008, 0.225207,
        -0.204477, -0.118362,
        -0.518564, -0.898180,
        -0.243605, -0.532652,
        -0.000266, -0.236263,
        0.517890, 0.299004,
        0.147031, 0.195748,
        -0.204743, 0.117901,
        -1.037129, -0.000000,
        -0.583093, -0.055358,
        -0.204743, -0.117901,
        0.517890, -0.299004,
        0.147031, -0.195748,
        -0.204743, -0.117901,
        -1.037129, 0.000000
        ],
        //wide rectangle
        vec![
            -RECT_W_H_RATIO, 1.0,
            -RECT_W_H_RATIO, -1.0,
            RECT_W_H_RATIO, -1.0,
            RECT_W_H_RATIO, 1.0,
            -RECT_W_H_RATIO, 1.0,
        ],
    ]
}
const RECT_W_H_RATIO: f32 = 3.0;
