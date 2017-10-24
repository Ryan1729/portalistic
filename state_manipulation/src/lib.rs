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
    let goal_nodes = vec![rng.gen()];

    let mut screens = vec![
        Screen {
            goals: vec![new_goal(&mut rng, &goal_nodes)],
            goal_nodes,
            ..Default::default()
        },
        Screen {
            goal_nodes: vec![rng.gen()],
            ..Default::default()
        },
    ];

    add_random_portal_pair(&mut rng, &mut screens, 0, 1);


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
        screens,
        screen_index: 0,
        portal_smell: 0,
        phase: Move,
    };

    state
}

fn new_goal(rng: &mut StdRng, goal_nodes: &Vec<GoalNode>) -> Goal {
    const GOAL_NODE_RADIUS: f32 = 1.0 / 16.0;

    if let Some(goal_node) = rng.choose(goal_nodes) {
        Goal {
            x: rng.gen_range(
                goal_node.x - GOAL_NODE_RADIUS,
                goal_node.x + GOAL_NODE_RADIUS,
            ),
            y: rng.gen_range(
                goal_node.y - GOAL_NODE_RADIUS,
                goal_node.y + GOAL_NODE_RADIUS,
            ),
        }
    } else {
        rng.gen()
    }
}

fn add_random_portal_pair(
    rng: &mut StdRng,
    screens: &mut Vec<Screen>,
    source_screen_index: ScreenIndex,
    target_screen_index: ScreenIndex,
) {
    let first_portal_spec = PortalSpec {
        x: rng.gen_range(-0.875, 0.875),
        y: rng.gen_range(-0.875, 0.875),
        screen_index: source_screen_index,
    };

    let second_portal_spec = PortalSpec {
        x: rng.gen_range(-0.875, 0.875),
        y: rng.gen_range(-0.875, 0.875),
        screen_index: target_screen_index,
    };

    add_portal_pair(screens, first_portal_spec, second_portal_spec)
}

fn add_portal_pair(screens: &mut Vec<Screen>, first: PortalSpec, second: PortalSpec) {
    match get_portals_mut_parts_pair(screens, first.screen_index, second.screen_index) {
        Pair::Both(first_portals, second_portals) => {
            //A `PortalSpec`'s `screen_index` indicates which screen to insert the portal into.
            //A `PortalTarget`'s `screen_index` field stores which screen it leads to.
            //The `target` field is the index in the other `portals` Vec where the other portal
            //will be placed.
            let first_target = PortalTarget {
                screen_index: second.screen_index,
                target: second_portals.len(),
            };
            let second_target = PortalTarget {
                screen_index: first.screen_index,
                target: first_portals.len(),
            };

            first_portals.push(Portal::from_spec(first, first_target));
            second_portals.push(Portal::from_spec(second, second_target));
        }
        Pair::One(portals) => {
            let len = portals.len();

            let first_target = PortalTarget {
                screen_index: second.screen_index,
                target: len + 1,
            };
            let second_target = PortalTarget {
                screen_index: first.screen_index,
                target: len,
            };

            portals.push(Portal::from_spec(first, first_target));
            portals.push(Portal::from_spec(second, second_target));
        }
        Pair::None => {}
    }
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
            Event::KeyDown(Keycode::P) => {
                //TODO allow selecting which screens portals will be placed on
                state.phase = PlaceFirstPortal((0, 1), state.screen_index);
            }
            Event::KeyDown(Keycode::G) => {
                fn new_goal_node(rng: &mut StdRng, goal_nodes: &Vec<GoalNode>) -> GoalNode {
                    let mut possible_goal_nodes: Vec<GoalNode> = Vec::new();

                    for _ in 0..8 {
                        possible_goal_nodes.push(rng.gen());
                    }

                    //If this O(n^2) part ends up too slow, then we can probably
                    //sample `goal_nodes` and pick the farthest away point from those
                    let distance_sq_sums = possible_goal_nodes.iter().map(|possible_goal_node| {
                        goal_nodes.iter().fold(0.0, |acc, goal_node| {
                            acc
                                + get_euclidean_sq(
                                    (goal_node.x, goal_node.y),
                                    (possible_goal_node.x, possible_goal_node.y),
                                )
                        })
                    });

                    let mut max_so_far = 0.0;
                    let mut max_index = 0;
                    for (i, distance_sq_sum) in distance_sq_sums.enumerate() {
                        if distance_sq_sum >= max_so_far {
                            max_index = i;
                            max_so_far = distance_sq_sum;
                        }
                    }

                    possible_goal_nodes
                        .get(max_index)
                        .cloned()
                        .unwrap_or_else(|| rng.gen())
                }

                if let Some(screen) = state.screens.get_mut(state.screen_index) {
                    let new_node = new_goal_node(&mut state.rng, &screen.goal_nodes);

                    screen.goal_nodes.push(new_node);
                }
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
        PlaceFirstPortal((source_screen_index, target_screen_index), previous_screen_index) => {
            state.screen_index = source_screen_index;
            if background_button_outcome.clicked {
                let portal_spec = PortalSpec {
                    x: mouse_x,
                    y: mouse_y,
                    screen_index: source_screen_index,
                };

                state.phase =
                    PlaceSecondPortal(target_screen_index, portal_spec, previous_screen_index);
            }
        }
        PlaceSecondPortal(target_screen_index, first_portal_spec, previous_screen_index) => {
            state.screen_index = target_screen_index;

            if background_button_outcome.clicked {
                let second_portal_spec = PortalSpec {
                    x: mouse_x,
                    y: mouse_y,
                    screen_index: target_screen_index,
                };

                add_portal_pair(&mut state.screens, first_portal_spec, second_portal_spec);

                state.phase = Move;
                state.screen_index = previous_screen_index;
            }
        }
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
                    let (sx, sy) = (state.x, state.y);
                    if let Some(portal_target) =
                        get_portals_mut_parts(&mut state.screens, state.screen_index)
                            .and_then(|portals| overlapping_portal_target(&portals, sx, sy))
                    {
                        if let Some(portal) =
                            get_portals_mut_parts(&mut state.screens, portal_target.screen_index)
                                .and_then(|portals| portals.get(portal_target.target))
                        {
                            state.x = portal.x;
                            state.y = portal.y;
                            state.screen_index = portal_target.screen_index;

                            state.portal_smell += PORTAL_SMELL_NS_PER;
                        }
                    }
                }

                let mut need_new_goal = false;
                if let Some(screen) = state.screens.get_mut(state.screen_index) {
                    if let Some(i) = overlapping_goal_index(&screen.goals, state.x, state.y) {
                        screen.goals.swap_remove(i);


                        need_new_goal = true;
                    }
                }

                let screen_count = state.screens.len();
                if need_new_goal && screen_count > 0 {
                    if let Some(screen) =
                        state.screens.get_mut(state.rng.gen_range(0, screen_count))
                    {
                        screen
                            .goals
                            .push(new_goal(&mut state.rng, &screen.goal_nodes));
                    }
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

    if let Some(portals) = get_portals_mut(state) {
        for portal in portals.iter() {
            draw_portal(portal.x, portal.y);
        }
    }

    if let Some(goal_nodes) = get_goal_nodes(state) {
        for goal_node in goal_nodes.iter() {
            (p.draw_poly_with_matrix_and_colours)(
                mat4x4_mul(
                    &view,
                    &scale_translation(1.0 / 8.0, goal_node.x, goal_node.y),
                ),
                (0.0, 0.0, 0.0, 0.0),
                (192.0 / 255.0, 48.0 / 255.0, 48.0 / 255.0, 0.375),
                3,
                0,
            );
        }
    }

    if let Some(goals) = get_goals(state) {
        for goal in goals.iter() {
            (p.draw_poly_with_matrix_and_colours)(
                mat4x4_mul(&view, &scale_translation(1.0 / 16.0, goal.x, goal.y)),
                (83.0 / 255.0, 36.0 / 255.0, 36.0 / 255.0, 1.0),
                (192.0 / 255.0, 192.0 / 255.0, 48.0 / 255.0, 1.0),
                3,
                0,
            );
        }
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
        PlaceFirstPortal(_, _) | PlaceSecondPortal(_, _, _) => {
            draw_portal(mouse_x, mouse_y);
        }
        _ => {}
    }



    false
}

fn get_portals_mut(state: &mut State) -> Option<&mut Vec<Portal>> {
    get_portals_mut_parts(&mut state.screens, state.screen_index)
}

fn get_portals_mut_parts(
    screens: &mut Vec<Screen>,
    screen_index: ScreenIndex,
) -> Option<&mut Vec<Portal>> {
    screens
        .get_mut(screen_index)
        .map(|screen| &mut screen.portals)
}

fn get_portals_mut_parts_pair(
    screens: &mut Vec<Screen>,
    screen_index_1: ScreenIndex,
    screen_index_2: ScreenIndex,
) -> Pair<&mut Vec<Portal>> {
    index_twice(screens, screen_index_1, screen_index_2).map_into(|screen| &mut screen.portals)
}

fn get_goals(state: &mut State) -> Option<&Vec<Goal>> {
    get_goals_parts(&mut state.screens, state.screen_index)
}

fn get_goals_parts(screens: &mut Vec<Screen>, screen_index: ScreenIndex) -> Option<&Vec<Goal>> {
    screens.get_mut(screen_index).map(|screen| &screen.goals)
}

fn get_goal_nodes(state: &mut State) -> Option<&Vec<GoalNode>> {
    get_goal_nodes_parts(&mut state.screens, state.screen_index)
}

fn get_goal_nodes_parts(
    screens: &mut Vec<Screen>,
    screen_index: ScreenIndex,
) -> Option<&Vec<GoalNode>> {
    screens
        .get_mut(screen_index)
        .map(|screen| &screen.goal_nodes)
}

// from https://stackoverflow.com/a/30075629/4496839
enum Pair<T> {
    Both(T, T),
    One(T),
    None,
}

fn index_twice<T>(slc: &mut [T], a: usize, b: usize) -> Pair<&mut T> {
    if a == b {
        slc.get_mut(a).map_or(Pair::None, Pair::One)
    } else {
        if a >= slc.len() || b >= slc.len() {
            Pair::None
        } else {
            // safe because a, b are in bounds and distinct
            unsafe {
                let ar = &mut *(slc.get_unchecked_mut(a) as *mut _);
                let br = &mut *(slc.get_unchecked_mut(b) as *mut _);
                Pair::Both(ar, br)
            }
        }
    }
}

impl<T> Pair<T> {
    fn map_into<U, F>(self, f: F) -> Pair<U>
    where
        F: Fn(T) -> U,
    {
        match self {
            Pair::Both(t1, t2) => Pair::Both(f(t1), f(t2)),
            Pair::One(t) => Pair::One(f(t)),
            Pair::None => Pair::None,
        }
    }
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

fn overlapping_portal_target(portals: &Vec<Portal>, x: f32, y: f32) -> Option<PortalTarget> {
    //curently I'm not expecting more than  16-ish portals on a screen,
    //and eventually I expect screens to be separated out so O(N) will
    //probably be fine.

    let mut result = None;

    const MINIMUM_DISTANCE_SQ: f32 = 0.0005;

    let mut smallest_so_far = std::f32::INFINITY;

    for portal in portals.iter() {
        let distance_sq = get_euclidean_sq((x, y), (portal.x, portal.y));

        if distance_sq < MINIMUM_DISTANCE_SQ && MINIMUM_DISTANCE_SQ < smallest_so_far {
            result = Some(portal.target.clone());

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
