use super::*;
use crate::bunching::*;
use crate::interface::*;
use crate::utility::*;
use crate::card::*;
use std::mem::{self, MaybeUninit};
use std::u8;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Default)]
struct BuildTreeInfo {
    flop_index: usize,
    turn_index: usize,
    river_index: usize,
    num_storage: u64,
    num_storage_ip: u64,
    num_storage_chance: u64,
}

impl Game for PostFlopGame {
    type P = PostFlopPair;

    #[inline]
    fn root(&self) -> MutexGuardLike<<PostFlopPair as GamePair>::N> {
        self.node_arena[0].lock()
    }

    #[inline]
    fn num_private_hands(&self, player: usize) -> usize {
        self.private_cards[player].len()
    }

    #[inline]
    fn initial_weights(&self, player: usize) -> &[f32] {
        &self.initial_weights[player]
    }

    #[inline]
    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &<PostFlopPair as GamePair>::N,
        player: usize,
        cfreach: &[f32],
    ) {
        if self.bunching_num_dead_cards == 0 {
            self.evaluate_internal(result, node, player, cfreach);
        } else {
            self.evaluate_internal_bunching(result, node, player, cfreach);
        }
    }

    #[inline]
    fn chance_factor(&self, node: &<PostFlopPair as GamePair>::N) -> usize {
        if node.turn == NOT_DEALT {
            45 - self.bunching_num_dead_cards
        } else {
            44 - self.bunching_num_dead_cards
        }
    }

    #[inline]
    fn is_solved(&self) -> bool {
        self.state == State::Solved
    }

    #[inline]
    fn set_solved(&mut self) {
        self.state = State::Solved;
        let history = self.action_history.clone();
        self.apply_history(&history);
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.state == State::MemoryAllocated && self.storage_mode == BoardState::River
    }

    #[inline]
    fn is_raked(&self) -> bool {
        self.tree_config.rake_rate > 0.0 && self.tree_config.rake_cap > 0.0
    }

    #[inline]
    fn isomorphic_chances(&self, node: &<PostFlopPair as GamePair>::N) -> &[u8] {
        if node.turn == NOT_DEALT {
            &self.isomorphism_ref_turn
        } else {
            &self.isomorphism_ref_river[node.turn as usize]
        }
    }

    #[inline]
    fn isomorphic_swap(&self, node: &<PostFlopPair as GamePair>::N, index: usize) -> &[Vec<(u16, u16)>; 2] {
        if node.turn == NOT_DEALT {
            &self.isomorphism_swap_turn[self.isomorphism_card_turn[index] as usize & 3]
        } else {
            &self.isomorphism_swap_river[node.turn as usize & 3]
                [self.isomorphism_card_river[node.turn as usize & 3][index] as usize & 3]
        }
    }

    #[inline]
    fn is_compression_enabled(&self) -> bool {
        self.is_compression_enabled
    }

    
    fn cut_them_locks<T>(&self, locks: Vec<T>, player: usize) -> Vec<T> 
    where
        T: Copy
    {
        let pairs: Vec<usize> = self.private_cards[player].iter().map(|&(c1, c2)| card_pair_to_index(c1, c2)).collect();
        let action_num = locks.len() / RANGESIZE;
        
        let mut new_t = Vec::with_capacity(action_num * pairs.len());

        for action in 0..action_num {
            for pair in pairs.iter() {
                new_t.push(locks[(pair.clone() as usize) + action * RANGESIZE]);
            }
        }

        new_t
    }
}

impl PostFlopGame {
    /// Creates a new empty [`PostFlopGame`].
    ///
    /// Use of this method is strongly discouraged because an instance created by this method is
    /// invalid until [`update_config`] is called.
    /// Please use [`with_config`] instead whenever possible.
    ///
    /// [`update_config`]: #method.update_config
    /// [`with_config`]: #method.with_config
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`PostFlopGame`] with the specified configuration.
    #[inline]
    pub fn with_config(card_config: CardConfig, action_tree: ActionTree) -> Result<Self, String> {
        let mut game = Self::new();
        game.update_config(card_config, action_tree)?;
        Ok(game)
    }

    /// Updates the game configuration. The solved result will be lost.
    #[inline]
    pub fn update_config(
        &mut self,
        card_config: CardConfig,
        action_tree: ActionTree,
    ) -> Result<(), String> {
        self.state = State::ConfigError;

        if !action_tree.invalid_terminals().is_empty() {
            return Err("Invalid terminal is found in action tree".to_string());
        }

        self.card_config = card_config;
        (
            self.tree_config,
            self.added_lines,
            self.removed_lines,
            self.action_root,
        ) = action_tree.eject();

        self.check_card_config()?;
        self.init_card_fields();
        self.init_root()?;

        self.state = State::TreeBuilt;

        self.init_interpreter();
        self.reset_bunching_effect();

        Ok(())
    }

    /// Sets the bunching effect configuration.
    ///
    /// **Warning**: Enabling the bunching effect will significantly slow down the solving process.
    /// Specifically, the computational complexity of the terminal evaluation will increase from
    /// *O*(#(OOP private hands) + #(IP private hands)) to *O*(#(OOP private hands) * #(IP private
    /// hands)).
    #[inline]
    pub fn set_bunching_effect(&mut self, bunching_data: &BunchingData) -> Result<(), String> {
        if self.state <= State::Uninitialized {
            return Err("Game is not successfully initialized".to_string());
        }

        if !bunching_data.is_ready() {
            return Err("Bunching configuration is not ready".to_string());
        }

        let mut flop_sorted = self.card_config.flop;
        flop_sorted.sort_unstable();
        if flop_sorted != bunching_data.flop() {
            return Err("Flop cards do not match".to_string());
        }

        self.reset_bunching_effect();
        self.set_bunching_effect_internal(bunching_data)?;

        Ok(())
    }

    /// Resets the bunching effect configuration. The current node will also be reset to the root.
    #[inline]
    pub fn reset_bunching_effect(&mut self) {
        self.bunching_num_dead_cards = 0;
        self.bunching_num_combinations = 0.0;
        self.bunching_arena = Vec::new();
        self.bunching_strength = Vec::new();
        self.bunching_num_flop = Default::default();
        self.bunching_num_turn = Default::default();
        self.bunching_num_river = Default::default();
        self.bunching_coef_flop = Default::default();
        self.bunching_coef_turn = Default::default();
        self.back_to_root();
    }

    /// Obtains the card configuration.
    #[inline]
    pub fn card_config(&self) -> &CardConfig {
        &self.card_config
    }

    /// Obtains the tree configuration.
    #[inline]
    pub fn tree_config(&self) -> &TreeConfig {
        &self.tree_config
    }

    /// Obtains the added lines.
    #[inline]
    pub fn added_lines(&self) -> &[Vec<Action>] {
        &self.added_lines
    }

    /// Obtains the removed lines.
    #[inline]
    pub fn removed_lines(&self) -> &[Vec<Action>] {
        &self.removed_lines
    }

    /// Returns the card list of private hands of the given player.
    ///
    /// The returned list contains only card pairs with positive weight, i.e., card pairs with zero
    /// weight are excluded. The returned list is sorted as follows:
    ///
    /// - Each card pair has IDs in `(low_id, high_id)` order.
    /// - Card pairs are sorted in the lexicographic order.
    #[inline]
    pub fn private_cards(&self, player: usize) -> &[(Card, Card)] {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        &self.private_cards[player]
    }

    /// Returns the estimated memory usage in bytes (uncompressed, compressed).
    #[inline]
    pub fn memory_usage(&self) -> (u64, u64) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        let num_elements = 2 * self.num_storage + self.num_storage_ip + self.num_storage_chance;
        let uncompressed = 4 * num_elements + self.misc_memory_usage;
        let compressed = 2 * num_elements + self.misc_memory_usage;

        (uncompressed, compressed)
    }

    /// Returns the estimated additional memory usage in bytes when the bunching effect is enabled.
    #[inline]
    pub fn memory_usage_bunching(&self) -> u64 {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        self.memory_usage_bunching_internal()
    }

    /// Remove lines after building the `PostFlopGame` but before allocating memory.
    ///
    /// This allows the removal of chance-specific lines (e.g., remove overbets on board-pairing
    /// turns) which we cannot do while building an action tree.
    pub fn remove_lines(&mut self, lines: &[Vec<Action>]) -> Result<(), String> {
        if self.state <= State::Uninitialized {
            return Err("Game is not successfully initialized".to_string());
        } else if self.state >= State::MemoryAllocated {
            return Err("Game has already been allocated".to_string());
        }

        for line in lines {
            let mut root = self.root();
            let info = self.remove_line_recursive(&mut root, line)?;
            self.num_storage -= info.num_storage;
            self.num_storage_ip -= info.num_storage_ip;
            self.num_storage_chance -= info.num_storage_chance;
        }

        Ok(())
    }

    /// Returns whether the memory is allocated.
    ///
    /// If the memory is allocated, returns `Some(is_compression_enabled)`;
    /// otherwise, returns `None`.
    #[inline]
    pub fn is_memory_allocated(&self) -> Option<bool> {
        if self.state <= State::TreeBuilt {
            None
        } else {
            Some(self.is_compression_enabled)
        }
    }

    /// Allocates the memory.
    pub fn allocate_memory(&mut self, enable_compression: bool) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if self.state == State::MemoryAllocated
            && self.storage_mode == BoardState::River
            && self.is_compression_enabled == enable_compression
        {
            return;
        }

        let num_bytes = if enable_compression { 2 } else { 4 };
        if num_bytes * self.num_storage > isize::MAX as u64
            || num_bytes * self.num_storage_chance > isize::MAX as u64
        {
            panic!("Memory usage exceeds maximum size");
        }

        self.state = State::MemoryAllocated;
        self.is_compression_enabled = enable_compression;

        self.clear_storage();

        let storage_bytes = (num_bytes * self.num_storage) as usize;
        let storage_ip_bytes = (num_bytes * self.num_storage_ip) as usize;
        let storage_chance_bytes = (num_bytes * self.num_storage_chance) as usize;

        self.storage1 = vec![0; storage_bytes];
        self.storage2 = vec![0; storage_bytes];
        self.storage_ip = vec![0; storage_ip_bytes];
        self.storage_chance = vec![0; storage_chance_bytes];

        self.allocate_memory_nodes();

        self.storage_mode = BoardState::River;
        self.target_storage_mode = BoardState::River;
    }

    /// Checks the card configuration.
    pub(crate) fn check_card_config(&mut self) -> Result<(), String> {
        let config = &self.card_config;
        let (flop, turn, river) = (config.flop, config.turn, config.river);
        let range = &config.range;

        if flop.contains(&NOT_DEALT) {
            return Err("Flop cards not initialized".to_string());
        }

        if flop.iter().any(|&c| 52 <= c) {
            return Err(format!("Flop cards must be in [0, 52): flop = {flop:?}"));
        }

        if flop[0] == flop[1] || flop[0] == flop[2] || flop[1] == flop[2] {
            return Err(format!("Flop cards must be unique: flop = {flop:?}"));
        }

        if turn != NOT_DEALT {
            if 52 <= turn {
                return Err(format!("Turn card must be in [0, 52): turn = {turn}"));
            }

            if flop.contains(&turn) {
                return Err(format!(
                    "Turn card must be different from flop cards: turn = {turn}"
                ));
            }
        }

        if river != NOT_DEALT {
            if 52 <= river {
                return Err(format!("River card must be in [0, 52): river = {river}"));
            }

            if flop.contains(&river) {
                return Err(format!(
                    "River card must be different from flop cards: river = {river}"
                ));
            }

            if turn == river {
                return Err(format!(
                    "River card must be different from turn card: river = {river}"
                ));
            }

            if turn == NOT_DEALT {
                return Err(format!(
                    "River card specified without turn card: river = {river}"
                ));
            }
        }

        let expected_state = match (turn != NOT_DEALT, river != NOT_DEALT) {
            (false, _) => BoardState::Flop,
            (true, false) => BoardState::Turn,
            (true, true) => BoardState::River,
        };

        if self.tree_config.initial_state != expected_state {
            return Err(format!(
                "Invalid initial state of `tree_config`: expected = {:?}, actual = {:?}",
                expected_state, self.tree_config.initial_state
            ));
        }

        if range[0].is_empty() {
            return Err("OOP range is empty".to_string());
        }

        if range[1].is_empty() {
            return Err("IP range is empty".to_string());
        }

        if !range[0].is_valid() {
            return Err("OOP range is invalid (loaded broken data?)".to_string());
        }

        if !range[1].is_valid() {
            return Err("IP range is invalid (loaded broken data?)".to_string());
        }

        self.init_hands();
        self.num_combinations = 0.0;

        for (&(c1, c2), &w1) in self.private_cards[0]
            .iter()
            .zip(self.initial_weights[0].iter())
        {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            for (&(c3, c4), &w2) in self.private_cards[1]
                .iter()
                .zip(self.initial_weights[1].iter())
            {
                let ip_mask: u64 = (1 << c3) | (1 << c4);
                if oop_mask & ip_mask == 0 {
                    self.num_combinations += w1 as f64 * w2 as f64;
                }
            }
        }

        if self.num_combinations == 0.0 {
            return Err("Valid card assignment does not exist".to_string());
        }

        Ok(())
    }

    /// Initializes fields `initial_weights` and `private_cards`.
    #[inline]
    fn init_hands(&mut self) {
        let config = &self.card_config;
        let (flop, turn, river) = (config.flop, config.turn, config.river);
        let range = &config.range;

        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        if turn != NOT_DEALT {
            board_mask |= 1 << turn;
        }
        if river != NOT_DEALT {
            board_mask |= 1 << river;
        }

        for player in 0..2 {
            let (hands, weights) = range[player].get_hands_weights(board_mask);
            self.initial_weights[player] = weights;
            self.private_cards[player] = hands;
        }
    }

    /// Initializes fields related to cards.
    pub(super) fn init_card_fields(&mut self) {
        for player in 0..2 {
            let same_hand_index = &mut self.same_hand_index[player];
            same_hand_index.clear();

            let player_hands = &self.private_cards[player];
            let opponent_hands = &self.private_cards[player ^ 1];
            for hand in player_hands {
                same_hand_index.push(
                    opponent_hands
                        .binary_search(hand)
                        .map_or(u16::MAX, |i| i as u16),
                );
            }
        }

        (
            self.valid_indices_flop,
            self.valid_indices_turn,
            self.valid_indices_river,
        ) = self.card_config.valid_indices(&self.private_cards);

        self.hand_strength = self.card_config.hand_strength(&self.private_cards);

        (
            self.isomorphism_ref_turn,
            self.isomorphism_card_turn,
            self.isomorphism_swap_turn,
            self.isomorphism_ref_river,
            self.isomorphism_card_river,
            self.isomorphism_swap_river,
        ) = self.card_config.isomorphism(&self.private_cards);

        
        println!("{:?}", self.isomorphism_card_turn);
        println!("{:?}", self.isomorphism_card_river);
    }

    /// Initializes the root node of game tree.
    fn init_root(&mut self) -> Result<(), String> {
        let num_nodes = self.count_num_nodes();
        let total_num_nodes = num_nodes[0] + num_nodes[1] + num_nodes[2];

        let mut buffer = BufferContainer {
            package_buffer: vec![vec![] as Vec<PackagedAction>; total_num_nodes as usize]
        };

        if total_num_nodes > u32::MAX as u64
            || mem::size_of::<PostFlopNode>() as u64 * total_num_nodes > isize::MAX as u64
        {
            return Err("Too many nodes".to_string());
        }

        self.num_nodes = num_nodes;
        self.node_arena = (0..total_num_nodes)
            .map(|_| MutexLike::new(PostFlopNode::default()))
            .collect::<Vec<_>>();
        self.clear_storage();

        let mut info = BuildTreeInfo {
            turn_index: num_nodes[0] as usize,
            river_index: (num_nodes[0] + num_nodes[1]) as usize,
            ..Default::default()
        };

        match self.tree_config.initial_state {
            BoardState::Flop => info.flop_index += 1,
            BoardState::Turn => info.turn_index += 1,
            BoardState::River => info.river_index += 1,
        }

        let mut root = self.node_arena[0].lock();
        root.turn = self.card_config.turn;
        root.river = self.card_config.river;

        self.build_tree_recursive(0, &self.action_root.lock(), &mut info, &mut buffer);

        self.num_storage = info.num_storage;
        self.num_storage_ip = info.num_storage_ip;
        self.num_storage_chance = info.num_storage_chance;

        self.lock_them_nodes(&buffer);

        self.misc_memory_usage = self.memory_usage_internal();

        Ok(())
    }

    /// Initializes the interpreter.
    #[inline]
    pub(super) fn init_interpreter(&mut self) {
        let vecs = [
            vec![0.0; self.num_private_hands(0)],
            vec![0.0; self.num_private_hands(1)],
        ];

        self.weights = vecs.clone();
        self.normalized_weights = vecs.clone();
        self.cfvalues_cache = vecs;
    }

    /// Clears the storage.
    #[inline]
    fn clear_storage(&mut self) {
        self.storage1 = Vec::new();
        self.storage2 = Vec::new();
        self.storage_ip = Vec::new();
        self.storage_chance = Vec::new();
    }

    #[inline]
    fn clear_locks(&mut self) {
        self.rstorage = MutexLike::new(Vec::new());
        self.lstorage = MutexLike::new(Vec::new());
        self.mrstorage = MutexLike::new(Vec::new());
        self.mlstorage = MutexLike::new(Vec::new());
    }

    /// Counts the number of nodes in the game tree.
    #[inline]
    fn count_num_nodes(&self) -> [u64; 3] {
        let (turn_coef, river_coef) = match (self.card_config.turn, self.card_config.river) {
            (NOT_DEALT, _) => {
                let mut river_coef = 0;
                let flop = self.card_config.flop;
                let skip_cards = &self.isomorphism_card_turn;
                let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
                let skip_mask: u64 = skip_cards.iter().map(|&card| 1 << card).sum();
                for turn in 0..52 {
                    if (1 << turn) & (flop_mask | skip_mask) == 0 {
                        river_coef += 48 - self.isomorphism_card_river[turn & 3].len();
                    }
                }
                (49 - self.isomorphism_card_turn.len(), river_coef)
            }
            (turn, NOT_DEALT) => (1, 48 - self.isomorphism_card_river[turn as usize & 3].len()),
            _ => (0, 1),
        };

        let num_action_nodes = count_num_action_nodes(&self.action_root.lock());

        [
            num_action_nodes[0],
            num_action_nodes[1] * turn_coef as u64,
            num_action_nodes[2] * river_coef as u64,
        ]
    }

    /// Computes the memory usage of this struct.
    #[inline]
    fn memory_usage_internal(&self) -> u64 {
        // untracked: tree_config, action_root

        let mut memory_usage = mem::size_of::<Self>() as u64;

        memory_usage += vec_memory_usage(&self.added_lines);
        memory_usage += vec_memory_usage(&self.removed_lines);
        for line in &self.added_lines {
            memory_usage += vec_memory_usage(line);
        }
        for line in &self.removed_lines {
            memory_usage += vec_memory_usage(line);
        }

        memory_usage += vec_memory_usage(&self.valid_indices_turn);
        memory_usage += vec_memory_usage(&self.valid_indices_river);
        memory_usage += vec_memory_usage(&self.hand_strength);
        memory_usage += vec_memory_usage(&self.isomorphism_ref_turn);
        memory_usage += vec_memory_usage(&self.isomorphism_card_turn);
        memory_usage += vec_memory_usage(&self.isomorphism_ref_river);

        memory_usage += vec_memory_usage(unsafe {self.rstorage.yoink()});
        memory_usage += vec_memory_usage(unsafe {self.lstorage.yoink()});
        memory_usage += vec_memory_usage(unsafe {self.mrstorage.yoink()});
        memory_usage += vec_memory_usage(unsafe {self.mlstorage.yoink()});

        for refs in &self.isomorphism_ref_river {
            memory_usage += vec_memory_usage(refs);
        }

        for cards in &self.isomorphism_card_river {
            memory_usage += vec_memory_usage(cards);
        }

        for player in 0..2 {
            memory_usage += vec_memory_usage(&self.initial_weights[player]);
            memory_usage += vec_memory_usage(&self.private_cards[player]);
            memory_usage += vec_memory_usage(&self.same_hand_index[player]);
            memory_usage += vec_memory_usage(&self.valid_indices_flop[player]);
            for indices in &self.valid_indices_turn {
                memory_usage += vec_memory_usage(&indices[player]);
            }
            for indices in &self.valid_indices_river {
                memory_usage += vec_memory_usage(&indices[player]);
            }
            for strength in &self.hand_strength {
                memory_usage += vec_memory_usage(&strength[player]);
            }
            for swap in &self.isomorphism_swap_turn {
                memory_usage += vec_memory_usage(&swap[player]);
            }
            for swap_list in &self.isomorphism_swap_river {
                for swap in swap_list {
                    memory_usage += vec_memory_usage(&swap[player]);
                }
            }
        }

        memory_usage += vec_memory_usage(&self.node_arena);

        memory_usage
    }

    /// Builds the game tree recursively.
    fn build_tree_recursive(
        &self,
        node_index: usize,
        action_node: &ActionTreeNode,
        info: &mut BuildTreeInfo,
        buffer: &mut BufferContainer
    )
    {
        let mut node = self.node_arena[node_index].lock();
        node.player = action_node.player;
        node.amount = action_node.amount;
        buffer.package_buffer[node_index] = action_node.actions.clone();

        if node.is_terminal() {
            return;
        }

        if node.is_chance() {
            self.push_chances(node_index, info);
            for action_index in 0..node.num_actions() {
                let child_index = node_index + node.children_offset as usize + action_index;
                self.build_tree_recursive(child_index, &action_node.children[0].lock(), info, buffer);
            }
        } else {
            self.push_actions(node_index, action_node, info);
            for action_index in 0..node.num_actions() {
                let child_index = node_index + node.children_offset as usize + action_index;
                self.build_tree_recursive(
                    child_index,
                    &action_node.children[action_index].lock(),
                    info,
                    buffer
                );
            }
        }
    }

    // ONE THREAD AT A TIME PLEASE
    fn lock_them_nodes(&self, buffer: &BufferContainer)
    {
        const VERBOSE: bool = false;

        if VERBOSE { println!("lock_them_nodes: me: {:?}", unsafe { self.mrstorage.yoink() }); }

        for node_id in 0..self.node_arena.len()
        {
            let p_actions = buffer.package_buffer[node_id].clone();
            let mut node = self.node_arena[node_id].lock();
            
            if p_actions.len() != 0
            {
                push_nodelocks(&mut node, self, p_actions);
            }
        }

        if VERBOSE { println!("lock_them_nodes: me: {:?}", unsafe { self.mrstorage.yoink() }); }

        if VERBOSE { println!("lock_them_nodes: NODELOCKING DATA"); }

        if VERBOSE { println!("lock_them_nodes: {}", unsafe { self.mrstorage.yoink().len() }); }
        if VERBOSE { println!("lock_them_nodes: {}", unsafe { self.rstorage.yoink().len() }); }

        if VERBOSE { println!("lock_them_nodes: {:?}", unsafe { self.rstorage.yoink() }); }
    }

    /// Pushes the chance actions to the `node`.
    fn push_chances(&self, node_index: usize, info: &mut BuildTreeInfo) {
        let mut node = self.node_arena[node_index].lock();
        let flop = self.card_config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        // deal turn
        if node.turn == NOT_DEALT {
            let skip_cards = &self.isomorphism_card_turn;
            let skip_mask: u64 = skip_cards.iter().map(|&card| 1 << card).sum();

            node.children_offset = (info.turn_index - node_index) as u32;
            for card in 0..52 {
                if (1 << card) & (flop_mask | skip_mask) == 0 {
                    node.num_children += 1;
                    let mut child = node.children().last().unwrap().lock();
                    child.prev_action = Action::Chance(card);
                    child.turn = card;
                }
            }

            info.turn_index += node.num_children as usize;
        }
        // deal river
        else {
            let turn_mask = flop_mask | (1 << node.turn);
            let skip_cards = &self.isomorphism_card_river[node.turn as usize & 3];
            let skip_mask: u64 = skip_cards.iter().map(|&card| 1 << card).sum();

            node.children_offset = (info.river_index - node_index) as u32;
            for card in 0..52 {
                if (1 << card) & (turn_mask | skip_mask) == 0 {
                    node.num_children += 1;
                    let mut child = node.children().last().unwrap().lock();
                    child.prev_action = Action::Chance(card);
                    child.turn = node.turn;
                    child.river = card;
                }
            }

            info.river_index += node.num_children as usize;
        }

        node.num_elements = node
            .cfvalue_storage_player()
            .map_or(0, |player| self.num_private_hands(player)) as u32;

        info.num_storage_chance += node.num_elements as u64;
    }

    /// Pushes the actions to the `node`.
    fn push_actions(
        &self,
        node_index: usize,
        action_node: &ActionTreeNode,
        info: &mut BuildTreeInfo,
    ) {
        let mut node = self.node_arena[node_index].lock();

        let street = match (node.turn, node.river) {
            (NOT_DEALT, _) => BoardState::Flop,
            (_, NOT_DEALT) => BoardState::Turn,
            _ => BoardState::River,
        };

        let base = match street {
            BoardState::Flop => &mut info.flop_index,
            BoardState::Turn => &mut info.turn_index,
            BoardState::River => &mut info.river_index,
        };

        node.children_offset = (*base - node_index) as u32;
        node.num_children = action_node.children.len() as u16;
        *base += node.num_children as usize;

        for (child, p_action) in node.children().iter().zip(action_node.actions.iter()) {
            let mut child = child.lock();
            child.prev_action = (*p_action).action;
            child.turn = node.turn;
            child.river = node.river;
        }

        let num_private_hands = self.num_private_hands(node.player as usize);
        node.num_elements = (node.num_actions() * num_private_hands) as u32;
        node.num_elements_ip = match node.prev_action {
            Action::None | Action::Chance(_) => self.num_private_hands(PLAYER_IP as usize) as u16,
            _ => 0,
        };

        info.num_storage += node.num_elements as u64;
        info.num_storage_ip += node.num_elements_ip as u64;
    }

    /// Sets the bunching effect.
    fn set_bunching_effect_internal(&mut self, bunching_data: &BunchingData) -> Result<(), String> {
        self.bunching_num_dead_cards = bunching_data.fold_ranges().len() * 2;
        let mut arena = vec![0.0]; // store dummy element

        // hand strength
        self.bunching_strength = self
            .hand_strength
            .iter()
            .map(|strength| {
                if strength[0].is_empty() {
                    return [Vec::new(), Vec::new()];
                }

                let mut ret = [
                    vec![0; self.num_private_hands(0)],
                    vec![0; self.num_private_hands(1)],
                ];

                for player in 0..2 {
                    let len = strength[player].len();
                    for &item in &strength[player][1..len - 1] {
                        ret[player][item.index as usize] = item.strength;
                    }
                }

                ret
            })
            .collect();

        // flop num combinations
        if self.card_config.turn == NOT_DEALT {
            for player in 0..2 {
                let player_cards = &self.private_cards[player];
                let opponent_cards = &self.private_cards[player ^ 1];
                let mut indices = Vec::with_capacity(player_cards.len());

                for &(c1, c2) in player_cards {
                    indices.push(arena.len());
                    let player_mask: u64 = (1 << c1) | (1 << c2);

                    for &(c3, c4) in opponent_cards {
                        let opponent_mask: u64 = (1 << c3) | (1 << c4);
                        if player_mask & opponent_mask != 0 {
                            arena.push(0.0);
                        } else {
                            let mask = player_mask | opponent_mask;
                            arena.push(bunching_data.result_4cards(mask));
                        }
                    }
                }

                if player == 0 {
                    self.bunching_num_combinations = arena.iter().fold(0.0, |a, &x| a + x as f64);
                    if self.bunching_num_combinations == 0.0 {
                        self.reset_bunching_effect();
                        return Err("Valid combination not found".to_string());
                    }
                }

                self.bunching_num_flop[player] = indices;
            }
        }

        let flop_mask: u64 = self.card_config.flop.iter().map(|&c| 1 << c).sum();
        let skip_turn_mask: u64 = self.isomorphism_card_turn.iter().map(|&c| 1 << c).sum();

        // turn num combinations
        if self.card_config.river == NOT_DEALT {
            for player in 0..2 {
                let player_cards = &self.private_cards[player];
                let opponent_cards = &self.private_cards[player ^ 1];

                let buf = into_par_iter(0..52)
                    .map(|turn| {
                        let bit_turn: u64 = 1 << turn;
                        if bit_turn & (flop_mask | skip_turn_mask) != 0
                            || (self.card_config.turn != NOT_DEALT
                                && self.card_config.turn != turn as Card)
                        {
                            return Vec::new();
                        }

                        let mut outer = Vec::with_capacity(player_cards.len());

                        for &(c1, c2) in player_cards {
                            let player_mask: u64 = (1 << c1) | (1 << c2);
                            if player_mask & bit_turn != 0 {
                                outer.push(Vec::new());
                                continue;
                            }

                            let mut inner = Vec::with_capacity(opponent_cards.len());
                            for &(c3, c4) in opponent_cards {
                                let opponent_mask: u64 = (1 << c3) | (1 << c4);
                                if (player_mask | bit_turn) & opponent_mask != 0 {
                                    inner.push(0.0);
                                } else {
                                    let mask = player_mask | opponent_mask | bit_turn;
                                    inner.push(bunching_data.result_5cards(mask));
                                }
                            }

                            outer.push(inner);
                        }

                        outer
                    })
                    .collect::<Vec<_>>();

                self.bunching_num_turn[player] = Self::push_vec_to_arena_f32(&mut arena, buf);

                if self.card_config.turn != NOT_DEALT && player == 0 {
                    self.bunching_num_combinations = arena.iter().fold(0.0, |a, &x| a + x as f64);
                    if self.bunching_num_combinations == 0.0 {
                        self.reset_bunching_effect();
                        return Err("Valid combination not found".to_string());
                    }
                }
            }
        }

        let is_board_possible = |turn: Card, river: Card| {
            let bit_turn: u64 = 1 << turn;
            let bit_river: u64 = 1 << river;
            let iso_card = &self.isomorphism_card_river[turn as usize & 3];

            bit_turn & (flop_mask | skip_turn_mask) == 0
                && bit_river & flop_mask == 0
                && !iso_card.contains(&river)
                && (self.card_config.turn == NOT_DEALT || self.card_config.turn == turn)
                && (self.card_config.river == NOT_DEALT || self.card_config.river == river)
        };

        // river num combinations
        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let opponent_cards = &self.private_cards[player ^ 1];

            let buf = into_par_iter(0..52 * 51 / 2)
                .map(|index| {
                    let (board1, board2) = index_to_card_pair(index);
                    if !is_board_possible(board1, board2) && !is_board_possible(board2, board1) {
                        return Vec::new();
                    }

                    let board_mask: u64 = (1 << board1) | (1 << board2);
                    let mut outer = Vec::with_capacity(player_cards.len());

                    for &(c1, c2) in player_cards {
                        let player_mask: u64 = (1 << c1) | (1 << c2);
                        if player_mask & board_mask != 0 {
                            outer.push(Vec::new());
                            continue;
                        }

                        let mut inner = Vec::with_capacity(opponent_cards.len());
                        for &(c3, c4) in opponent_cards {
                            let opponent_mask: u64 = (1 << c3) | (1 << c4);
                            if (player_mask | board_mask) & opponent_mask != 0 {
                                inner.push(0.0);
                            } else {
                                let mask = player_mask | opponent_mask | board_mask;
                                inner.push(bunching_data.result_6cards(mask));
                            }
                        }

                        outer.push(inner);
                    }

                    outer
                })
                .collect::<Vec<_>>();

            self.bunching_num_river[player] = Self::push_vec_to_arena_f32(&mut arena, buf);

            if self.card_config.river != NOT_DEALT && player == 0 {
                self.bunching_num_combinations = arena.iter().fold(0.0, |a, &x| a + x as f64);
                if self.bunching_num_combinations == 0.0 {
                    self.reset_bunching_effect();
                    return Err("Valid combination not found".to_string());
                }
            }
        }

        if self.card_config.river != NOT_DEALT {
            self.bunching_arena = arena;
            self.assign_zero_weights();
            return Ok(());
        }

        // turn equity coefficients
        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let opponent_cards = &self.private_cards[player ^ 1];
            let player_len = player_cards.len();
            let opponent_len = opponent_cards.len();

            let buf = into_par_iter(0..52)
                .map(|turn| {
                    let bit_turn: u64 = 1 << turn;
                    if bit_turn & (flop_mask | skip_turn_mask) != 0
                        || (self.card_config.turn != NOT_DEALT
                            && self.card_config.turn != turn as Card)
                    {
                        return Vec::new();
                    }

                    let mut outer = Vec::with_capacity(player_len);

                    for &(c1, c2) in player_cards {
                        let player_mask: u64 = (1 << c1) | (1 << c2);
                        if player_mask & bit_turn != 0 {
                            outer.push(Vec::new());
                        } else {
                            outer.push(vec![0.0; opponent_len]);
                        }
                    }

                    let mut children = Vec::with_capacity(48);
                    let iso_ref = &self.isomorphism_ref_river[turn];
                    let iso_card = &self.isomorphism_card_river[turn & 3];
                    let iso_swap = &self.isomorphism_swap_river[turn & 3];

                    for river in 0..52 {
                        let bit_river: u64 = 1 << river;
                        if bit_river & (flop_mask | bit_turn) != 0 {
                            continue;
                        }

                        let pos = iso_card.iter().position(|&c| c == river);
                        let (river_ref, swap_option) = if let Some(pos) = pos {
                            let child_index = iso_ref[pos] as usize;
                            let swap = &iso_swap[river as usize & 3];
                            (children[child_index], Some(swap))
                        } else {
                            children.push(river);
                            (river, None)
                        };

                        let player_swap = swap_option.map(|swap| {
                            let mut tmp = (0..player_len).collect::<Vec<_>>();
                            apply_swap(&mut tmp, &swap[player]);
                            tmp
                        });

                        let pair_index = card_pair_to_index(turn as Card, river_ref);
                        let arena_indices = &self.bunching_num_river[player][pair_index];
                        let player_strength = &self.bunching_strength[pair_index][player];
                        let opponent_strength = &self.bunching_strength[pair_index][player ^ 1];

                        for (i, inner) in outer.iter_mut().enumerate() {
                            let player_index = player_swap.as_ref().map_or(i, |map| map[i]);
                            let index = arena_indices[player_index];
                            let threshold = player_strength[player_index];

                            if index == 0 {
                                continue;
                            }

                            let mut tmp = (Vec::new(), Vec::new());
                            let slices = if let Some(swap) = swap_option {
                                tmp.0.extend_from_slice(&arena[index..index + opponent_len]);
                                tmp.1.extend_from_slice(opponent_strength);
                                apply_swap(&mut tmp.0, &swap[player ^ 1]);
                                apply_swap(&mut tmp.1, &swap[player ^ 1]);
                                (tmp.0.as_slice(), &tmp.1)
                            } else {
                                (&arena[index..index + opponent_len], opponent_strength)
                            };

                            inner.iter_mut().zip(slices.0).zip(slices.1).for_each(
                                |((dst, num), &strength)| {
                                    #[allow(clippy::comparison_chain)]
                                    if strength < threshold {
                                        *dst += *num as f64;
                                    } else if strength > threshold {
                                        *dst += -*num as f64;
                                    } else {
                                        *dst += 0.0;
                                    }
                                },
                            );
                        }
                    }

                    let num_possible_river = (44 - self.bunching_num_dead_cards) as f64;
                    outer.iter_mut().for_each(|inner| {
                        inner.iter_mut().for_each(|c| {
                            *c /= num_possible_river;
                        });
                    });

                    outer
                })
                .collect::<Vec<_>>();

            self.bunching_coef_turn[player] = Self::push_vec_to_arena_f64(&mut arena, buf);
        }

        if self.card_config.turn != NOT_DEALT {
            self.bunching_arena = arena;
            self.assign_zero_weights();
            return Ok(());
        }

        // flop equity coefficients
        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let opponent_cards = &self.private_cards[player ^ 1];
            let player_len = player_cards.len();
            let opponent_len = opponent_cards.len();

            let mut outer = vec![vec![0.0; opponent_len]; player_len];
            let mut children = Vec::with_capacity(49);

            for turn in 0..52 {
                let bit_turn: u64 = 1 << turn;
                if bit_turn & flop_mask != 0 {
                    continue;
                }

                let iso_card = &self.isomorphism_card_turn;
                let pos = iso_card.iter().position(|&c| c == turn as Card);

                let (turn_ref, swap_option) = if let Some(pos) = pos {
                    let child_index = self.isomorphism_ref_turn[pos] as usize;
                    let swap = &self.isomorphism_swap_turn[turn & 3];
                    (children[child_index], Some(swap))
                } else {
                    children.push(turn);
                    (turn, None)
                };

                let player_swap = swap_option.map(|swap| {
                    let mut tmp = (0..player_len).collect::<Vec<_>>();
                    apply_swap(&mut tmp, &swap[player]);
                    tmp
                });

                let arena_indices = &self.bunching_coef_turn[player][turn_ref];

                for (i, inner) in outer.iter_mut().enumerate() {
                    let player_index = player_swap.as_ref().map_or(i, |map| map[i]);
                    let index = arena_indices[player_index];
                    if index == 0 {
                        continue;
                    }

                    let mut tmp = Vec::new();
                    let slice = &arena[index..index + opponent_len];
                    let slice = if let Some(swap) = swap_option {
                        tmp.extend_from_slice(slice);
                        apply_swap(&mut tmp, &swap[player ^ 1]);
                        &tmp
                    } else {
                        slice
                    };

                    inner.iter_mut().zip(slice).for_each(|(dst, &num)| {
                        *dst += num as f64;
                    });
                }
            }

            let num_possible_turn = (45 - self.bunching_num_dead_cards) as f64;
            outer.iter_mut().for_each(|inner| {
                inner.iter_mut().for_each(|c| {
                    *c /= num_possible_turn;
                });
            });

            Self::push_vec_to_arena_f64(&mut arena, vec![outer])
                .into_iter()
                .for_each(|v| {
                    self.bunching_coef_flop[player] = v;
                });
        }

        self.bunching_arena = arena;
        self.assign_zero_weights();
        Ok(())
    }

    /// Sets the bunching effect.
    fn memory_usage_bunching_internal(&self) -> u64 {
        let mut ret = 4;

        let oop_len = self.num_private_hands(0);
        let ip_len = self.num_private_hands(1);

        // hand strength
        self.hand_strength.iter().for_each(|strength| {
            ret += mem::size_of::<[Vec<u16>; 2]>() as u64;
            if !strength[0].is_empty() {
                ret += 2 * (oop_len + ip_len) as u64;
            }
        });

        // flop num combinations / equity coefficients
        if self.card_config.turn == NOT_DEALT {
            ret += 2 * (oop_len * ip_len * mem::size_of::<usize>()) as u64;
            ret += 2 * 2 * 4 * (oop_len * ip_len) as u64;
        }

        let flop_mask: u64 = self.card_config.flop.iter().map(|&c| 1 << c).sum();
        let skip_turn_mask: u64 = self.isomorphism_card_turn.iter().map(|&c| 1 << c).sum();

        // turn num combinations / equity coefficients
        if self.card_config.river == NOT_DEALT {
            for player in 0..2 {
                let player_cards = &self.private_cards[player];
                let opponent_cards = &self.private_cards[player ^ 1];
                let player_len = player_cards.len();
                let opponent_len = opponent_cards.len();

                ret += 2 * 52 * mem::size_of::<Vec<usize>>() as u64;

                for turn in 0..52 {
                    let bit_turn: u64 = 1 << turn;
                    if bit_turn & (flop_mask | skip_turn_mask) != 0
                        || (self.card_config.turn != NOT_DEALT
                            && self.card_config.turn != turn as Card)
                    {
                        continue;
                    }

                    ret += 2 * (player_len * mem::size_of::<usize>()) as u64;

                    for &(c1, c2) in player_cards {
                        let player_mask: u64 = (1 << c1) | (1 << c2);
                        if player_mask & bit_turn == 0 {
                            ret += 2 * 4 * opponent_len as u64;
                        }
                    }
                }
            }
        }

        let is_board_possible = |turn: Card, river: Card| {
            let bit_turn: u64 = 1 << turn;
            let bit_river: u64 = 1 << river;
            let iso_card = &self.isomorphism_card_river[turn as usize & 3];

            bit_turn & (flop_mask | skip_turn_mask) == 0
                && bit_river & flop_mask == 0
                && !iso_card.contains(&river)
                && (self.card_config.turn == NOT_DEALT || self.card_config.turn == turn)
                && (self.card_config.river == NOT_DEALT || self.card_config.river == river)
        };

        // river num combinations
        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let opponent_cards = &self.private_cards[player ^ 1];
            let player_len = player_cards.len();
            let opponent_len = opponent_cards.len();

            ret += 52 * 51 / 2 * mem::size_of::<Vec<usize>>() as u64;

            for index in 0..52 * 51 / 2 {
                let (board1, board2) = index_to_card_pair(index);
                if !is_board_possible(board1, board2) && !is_board_possible(board2, board1) {
                    continue;
                }

                let board_mask: u64 = (1 << board1) | (1 << board2);
                ret += (player_len * mem::size_of::<usize>()) as u64;

                for &(c1, c2) in player_cards {
                    let player_mask: u64 = (1 << c1) | (1 << c2);
                    if player_mask & board_mask == 0 {
                        ret += 4 * opponent_len as u64;
                    }
                }
            }
        }

        ret
    }

    /// Pushes a nested `Vec` to an arena and returns the indices of the pushed elements.
    fn push_vec_to_arena_f32(arena: &mut Vec<f32>, vec: Vec<Vec<Vec<f32>>>) -> Vec<Vec<usize>> {
        let mut ret = Vec::with_capacity(vec.len());

        for outer in vec.into_iter() {
            let mut indices = Vec::with_capacity(outer.len());

            for inner in outer.into_iter() {
                if inner.is_empty() {
                    indices.push(0);
                } else {
                    indices.push(arena.len());
                    arena.extend(inner);
                }
            }

            ret.push(indices);
        }

        ret
    }

    /// Pushes a nested `Vec` to an arena and returns the indices of the pushed elements.
    fn push_vec_to_arena_f64(arena: &mut Vec<f32>, vec: Vec<Vec<Vec<f64>>>) -> Vec<Vec<usize>> {
        let mut ret = Vec::with_capacity(vec.len());

        for outer in vec.into_iter() {
            let mut indices = Vec::with_capacity(outer.len());

            for inner in outer.into_iter() {
                if inner.is_empty() {
                    indices.push(0);
                } else {
                    indices.push(arena.len());
                    arena.extend(inner.into_iter().map(|v| v as f32));
                }
            }

            ret.push(indices);
        }

        ret
    }

    /// Calculates the number of storage elements that will be removed.
    fn calculate_removed_line_info_recursive(node: &mut PostFlopNode, info: &mut BuildTreeInfo) {
        if node.is_terminal() {
            return;
        }

        if node.is_chance() {
            info.num_storage_chance += node.num_elements as u64;
            node.num_elements = 0;
        } else {
            info.num_storage += node.num_elements as u64;
            info.num_storage_ip += node.num_elements_ip as u64;
            node.num_elements = 0;
            node.num_elements_ip = 0;
        }

        for action in node.action_indices() {
            Self::calculate_removed_line_info_recursive(&mut node.play(action), info);
        }
    }

    /// Remove a line from a `PostFlopGame` tree.
    fn remove_line_recursive(
        &self,
        node: &mut PostFlopNode,
        line: &[Action],
    ) -> Result<BuildTreeInfo, String> {
        if line.is_empty() {
            return Err("Empty line".to_string());
        }

        if node.is_terminal() {
            return Err("Unexpected terminal node".to_string());
        }

        let action = line[0];
        let search_result = node
            .children()
            .binary_search_by(|child| child.lock().prev_action.cmp(&action));

        if search_result.is_err() {
            return Err(format!("Action does not exist: {action:?}"));
        }

        let index = search_result.unwrap();
        if line.len() > 1 {
            let result = self.remove_line_recursive(&mut node.children()[index].lock(), &line[1..]);
            return result;
        }

        if node.is_chance() {
            return Err("Cannot remove a line ending in a chance action".to_string());
        }

        if node.num_actions() <= 1 {
            return Err("Cannot remove the last action from a node".to_string());
        }

        // Remove action/children at index. To do this we must
        // 1. compute the storage space required by the tree rooted at action index
        // 2. remove children and actions
        // 3. re-define `num_elements` after we remove children and actions

        // STEP 1
        let mut info = BuildTreeInfo {
            num_storage: self.num_private_hands(node.player as usize) as u64,
            ..Default::default()
        };

        let mut node_to_remove = node.play(index);
        Self::calculate_removed_line_info_recursive(&mut node_to_remove, &mut info);

        // STEP 2
        let children = node.children();
        for i in index..node.num_children as usize - 1 {
            let mut x = children[i].lock();
            let mut y = children[i + 1].lock();
            mem::swap(&mut *x, &mut *y);
            if x.children_offset > 0 {
                x.children_offset += 1;
            }
        }
        node.num_children -= 1;

        // STEP 3
        node.num_elements -= self.num_private_hands(node.player as usize) as u32;

        Ok(info)
    }

    /// Allocates memory recursively.
    fn allocate_memory_nodes(&mut self) {
        let num_bytes = if self.is_compression_enabled { 2 } else { 4 };
        let mut action_counter = 0;
        let mut ip_counter = 0;
        let mut chance_counter = 0;

        for node in &self.node_arena {
            let mut node = node.lock();
            if node.is_terminal() {
                // do nothing
            } else if node.is_chance() {
                unsafe {
                    let ptr = self.storage_chance.as_mut_ptr();
                    node.storage1 = ptr.add(chance_counter);
                }
                chance_counter += num_bytes * node.num_elements as usize;
            } else {
                unsafe {
                    let ptr1 = self.storage1.as_mut_ptr();
                    let ptr2 = self.storage2.as_mut_ptr();
                    let ptr3 = self.storage_ip.as_mut_ptr();
                    node.storage1 = ptr1.add(action_counter);
                    node.storage2 = ptr2.add(action_counter);
                    node.storage3 = ptr3.add(ip_counter);
                }
                action_counter += num_bytes * node.num_elements as usize;
                ip_counter += num_bytes * node.num_elements_ip as usize;
            }
        }
    }
}

fn push_nodelocks (node: &mut MutexGuardLike<PostFlopNode>, game: &PostFlopGame, p_actions: Vec<PackagedAction>)
{
    const VERBOSE: bool = false;

    if VERBOSE { println!("push_nodelocks: Starting packing it up!"); }

    let mut r_storage = game.rstorage.lock();
    let mut l_storage = game.lstorage.lock();
    let mut mr_storage = game.mrstorage.lock();
    let mut ml_storage = game.mlstorage.lock();

    let mut mr_data: Vec<u32> = vec![]; // offsets for the ranges of one specific node
    let mut ml_data: Vec<u32> = vec![]; // same, but for limits

    let mut r_hashes = game.rhashes.lock();
    let mut l_hashes = game.lhashes.lock();

    for mut packaged_action in p_actions
    {
        if VERBOSE { println!("push_nodelocks: Packing a packaged action"); }

        let mut end_range: [f32; RANGESIZE] = [0.0; RANGESIZE];
        let mut end_limit: [i8; RANGESIZE] = [1; RANGESIZE];

        packaged_action.sort_rules();


        if packaged_action.lock_rules.is_some() && packaged_action.lock_rules.as_ref().unwrap().len() > 0
        {
            println!("push_nodelocks: Ruling the rule");

            let rule_num = packaged_action.lock_rules.as_ref().unwrap().len();
            let rules = packaged_action.lock_rules.as_ref().unwrap().clone();

            let mut board = vec![];
            board.extend(game.card_config.flop);
            if node.turn  != u8::MAX {board.push(node.turn );}
            if node.river != u8::MAX {board.push(node.river);}

            if rules[0].priority > 0
            {
                (end_range, end_limit) = apply_range(packaged_action.clone(), end_range, end_limit, [true; RANGESIZE]);
            }

            let mut rulepack: Vec<RuleLock> = vec![];
            let mut trvthgrid: [bool; RANGESIZE] = [true; RANGESIZE];

            for i in 0..rule_num
            {
                rulepack.push(rules[i].clone());

                if i != rule_num - 1 && rules[i + 1].priority == rules[i].priority
                {
                    continue
                }
                else
                {
                    for r in rulepack.iter()
                    {
                        let thisgrid = get_trvth_grid(r, &board);
                        
                        for i in 0..RANGESIZE
                        {
                            trvthgrid[i] = trvthgrid[i] && thisgrid[i];
                        }
                    }
                }

                if rules[i].priority == 0
                {
                    (end_range, end_limit) = apply_range(packaged_action.clone(), end_range, end_limit, trvthgrid);
                }
                else
                {
                    for pair in 0..RANGESIZE
                    {
                        if trvthgrid[pair] == true
                        {
                            end_range[pair] = rules[i].percentage;
                            end_limit[pair] = rules[i].limitation;
                        }
                    }
                }


                rulepack.clear();
                trvthgrid = [true; RANGESIZE];

                if i != rule_num-1 && rules[i].priority < 0 && rules[i + 1].priority == rules[i].priority
                {
                    (end_range, end_limit) = apply_range(packaged_action.clone(), end_range, end_limit, [true; RANGESIZE]);
                }
            }
        }
        else
        {
            (end_range, end_limit) = apply_range(packaged_action.clone(), end_range, end_limit, [true; RANGESIZE]);
        }
        


        let mut hasher: DefaultHasher;
        let unsigned_range: Vec<u32> = end_range.iter().map(|f| *f as u32).collect();

        hasher = DefaultHasher::new();
        unsigned_range.hash(&mut hasher);
        let range_hash = hasher.finish();
        
        if VERBOSE { println!("my_end_range: range hash: {:}", range_hash); }

        hasher = DefaultHasher::new();
        end_limit.hash(&mut hasher);
        let limit_hash = hasher.finish();

        let mut r_loc: isize = -1;
        let mut l_loc: isize = -1;

        for t in 0..r_hashes.len()
        {
            if r_hashes[t].0 == range_hash && l_hashes[t].0 == limit_hash
            {
                r_loc = r_hashes[t].1 as isize;
                l_loc = l_hashes[t].1 as isize;
                break;
            }
        }

        if r_loc == -1
        {
            r_loc = r_storage.len() as isize;
            l_loc = l_storage.len() as isize;

            r_hashes.push((range_hash, r_loc as usize));
            l_hashes.push((limit_hash, l_loc as usize));

            r_storage.extend(end_range);
            l_storage.extend(end_limit);
        }

        mr_data.push(r_loc as u32);
        ml_data.push(l_loc as u32);

        if VERBOSE { println!("push_nodelocks: Action packed"); }
    }
    
    if VERBOSE { println!("my_end_range: mr_data: {:?}", mr_data); }
    if VERBOSE { println!("my_end_range: ml_data: {:?}", ml_data); }
    
    // saving the offset packages into mem megastorages

    let m_loc = mr_storage.len(); // they will be exactly the same due to how antipairs are implemented
    
    if VERBOSE { println!("my_end_range: m_loc: {:?}", m_loc); }

    mr_storage.extend(mr_data);
    ml_storage.extend(ml_data);
    
    // saving the pointer to node-related offset packages into the node itself

    node.mstorage_offset = m_loc as u32; // the previous ver was a terrible idea 
    // println!("my_end_range: mstorage offset: {:?}", node.mstorage_offset);
    
    if VERBOSE { println!("push_nodelocks: Finished here"); }
    
    fn apply_range (p_actions: PackagedAction, mut end_range: [f32; RANGESIZE], mut end_limit: [i8; RANGESIZE], trvthgrid: [bool; RANGESIZE]) -> ([f32; RANGESIZE], [i8; RANGESIZE])
    {
        const VERBOSE: bool = false;
        const RANGEEMPTY: [f32; 13*13] = [0.0; 13*13];

        if !p_actions.lock_range.is_none() 
        {
            let lock_range = p_actions.lock_range.unwrap().clone();
            let lock_limit = p_actions.lock_limit.unwrap().clone();

            if lock_range != RANGEEMPTY && VERBOSE
            {
                if VERBOSE {println!("apply_range: non-empty range found");}
            }

            for i in 0..13 { for j in 0..13
            {
                if i > j
                {
                    for suit_i in 0..4 { for suit_j in 0..4
                    {
                        // these are some nested loops aren't they

                        if suit_i == suit_j
                        {
                            continue
                        }
                        else
                        {
                            let card1 = (12 - i) * 4 + suit_i;
                            let card2 = (12 - j) * 4 + suit_j;

                            if VERBOSE {println!("apply_range: offsuit: card1: {:?}, card2: {:?}", card1, card2);}

                            let index = card_pair_to_index(card1 as Card, card2 as Card);
                            if trvthgrid[index] == true && !(lock_range[i * 13 + j] == 0.0 && lock_limit[i * 13 + j] == 1)
                            {
                                end_range[index] = lock_range[i * 13 + j];
                                end_limit[index] = lock_limit[i * 13 + j];
                            }
                        }
                    }}
                } 
                else if i < j
                {
                    for suit in 0..4
                    {
                        let card1 = (12 - i) * 4 + suit;
                        let card2 = (12 - j) * 4 + suit;

                        if VERBOSE {println!("apply_range: suited: card1: {:?}, card2: {:?}", card1, card2);}
                        
                        let index = card_pair_to_index(card1 as Card, card2 as Card);
                        if trvthgrid[index] == true && !(lock_range[i * 13 + j] == 0.0 && lock_limit[i * 13 + j] == 1)
                        {
                            end_range[index] = lock_range[i * 13 + j];
                            end_limit[index] = lock_limit[i * 13 + j];
                        }
                    }
                }
                else
                {
                    for suit_i in 0..4 { for suit_j in 0..4
                    {
                        // more nested loopz

                        if suit_i <= suit_j
                        {
                            continue
                        }
                        else
                        {
                            let card1 = (12 - i) * 4 + suit_i;
                            let card2 = (12 - j) * 4 + suit_j;

                            if VERBOSE {println!("apply_range: pair: card1: {:?}, card2: {:?}", card1, card2);}

                            let index = card_pair_to_index(card1 as Card, card2 as Card);
                            if trvthgrid[index] == true && !(lock_range[i * 13 + j] == 0.0 && lock_limit[i * 13 + j] == 1)
                            {
                                end_range[index] = lock_range[i * 13 + j];
                                end_limit[index] = lock_limit[i * 13 + j];
                            }
                        }
                    }}
                }
            } }
            
            println!("apply_range: result range: {:?}", end_range);

            if lock_range != RANGEEMPTY && VERBOSE
            {
                println!("apply_range: result range: {:?}", end_range);
            }
        }

        (end_range, end_limit)
    }

    fn get_trvth_grid (lock: &RuleLock, board: &Vec<Card>) -> [bool; RANGESIZE]
    {
        const VERBOSE: bool = false;

        let rule_type = lock.rule_type;
        let mut trvthgrid: [bool; RANGESIZE] = [false; RANGESIZE];

        for i in 0..RANGESIZE
        {
            if rule_type.0 == 0 // hand class processing

            {
                let data = evaluate_class(&index_to_card_pair(i));

                if rule_type.1 == data
                {
                    trvthgrid[i] = true;
                }
            }

            else if rule_type.0 == 1 // hand rank

            {
                let hand = index_to_card_pair(i);
                let rank: u8;
                
                if hand.0 > hand.1 // should always be true but im paranoid
                {
                    rank = hand.0 >> 2;
                }
                else
                {
                    rank = hand.1 >> 2;
                }

                if rule_type.1 == 0 && rank < rule_type.2
                {
                    trvthgrid[i] = true;
                }
                else if rule_type.1 == 1 && rank == rule_type.2
                {
                    trvthgrid[i] = true;
                }
                else if rule_type.1 == 2 && rank > rule_type.2
                {
                    trvthgrid[i] = true;
                }
                
            }

            else if rule_type.0 == 2 || rule_type.0 == 3 // made hand processing

            {
                let data = evaluate_rank(&index_to_card_pair(i), board);

                const IS_MANUAL: [bool; 10] = [true, true, false, false, true, true, true, false, false, false]; // gahh
                
                if rule_type.0 == 3 && data.0 > rule_type.1
                {
                    trvthgrid[i] = true;
                    continue;
                }

                if rule_type.1 == data.0
                {
                    if IS_MANUAL[data.0 as usize] // manual
                    {
                        if rule_type.2 == 0
                        {
                            trvthgrid[i] = true;
                        }
                        else
                        {
                            if rule_type.0 == 3 && data.1 < rule_type.2 // they just rank in the opposite order
                            {
                                trvthgrid[i] = true;
                            }
                            else
                            {
                                trvthgrid[i] = data.1 == rule_type.2;
                            }
                        }
                    }
                    else // autotrue
                    {
                        trvthgrid[i] = true;
                    }
                }
                
                
            }
            else if rule_type.0 == 3

            {
                // draw processing
            }

            else if rule_type.0 == 4

            {
                // board texture processing
            }
        }

        trvthgrid
    }


}

/// made hand eval
fn evaluate_rank (hand: &(Card, Card), board: &Vec<Card>) -> (u8, u8)
{
    let mut cards = vec![hand.0, hand.1];
    cards.extend(board);

    let board_sorted = {
        let mut tmp = board.clone();
        tmp.sort_unstable_by(|a, b| b.cmp(&a));
        tmp
    }; // hypothetically it should be kinda sorted but without sorts it all is going to implode so can't risk it

    let ret: (u8, u8); // hand type, related data

    let repeats_data = evaluate_repeats(hand, &board_sorted);
    let straight_data = evaluate_straight(hand, &board_sorted);
    let flush_data = evaluate_flush(hand, &board_sorted);

    let is_straight_flush = humble_straight_flush(hand, &board_sorted); // who cares about straight flush ranks, also just in case you can't check for straight flushes with the other functions as you having straight and flush doesn't necessarily mean you have a straight flush

    if is_straight_flush
    {
        ret = (9, u8::MAX); // max for the sake of flex
    }
    else if repeats_data.0 == 6 // four of a kind
    {
        ret = (8, repeats_data.1);
    }
    else if repeats_data.0 == 5 // full house
    {
        ret = (7, repeats_data.1);
    }
    else if flush_data.0 == true // flush, obviously
    {
        ret = (6, flush_data.1);
    }
    else if straight_data.0 != 0 // straight, obviously
    {
        ret = (5, straight_data.1);
    }
    else if repeats_data.0 == 4 // set
    {
        ret = (4, repeats_data.1);
    }
    else if repeats_data.0 == 3 // three of a kind, trips
    {
        ret = (3, repeats_data.1);
    }
    else if repeats_data.0 == 2 // two pair
    {
        ret = (2, repeats_data.1);
    }
    else if repeats_data.0 == 1 // pair
    {
        ret = (1, repeats_data.1);
    }
    else // brokie high card T_T
    {
        ret = (0, evaluate_brokie(hand, &board_sorted));
    }
    
    // FUNCS     
    
    /// high card evaluator
    fn evaluate_brokie (hand: &(Card, Card), board: &Vec<Card>) -> u8
    {
        let mut cards = vec![hand.0, hand.1];
        cards.extend(board);

        let board_deranked = derank(board);

        let mut ret: u8 = 1; // how top is the high card

        let topcard = if hand.0 > hand.1 { hand.0 >> 2 } else { hand.1 >> 2 };

        for checkrank in (0..13).rev()
        {
            if board_deranked.contains(&checkrank)
            {
                continue;
            }
            
            if topcard == checkrank
            {
                break;
            }

            ret += 1;

            if ret == 3
            {
                break;
            }
        }

        ret
    }

    fn evaluate_repeats (hand: &(Card, Card), board: &Vec<Card>) -> (u8, u8)
    {
        let mut cards = vec![hand.0, hand.1];
        cards.extend(board);

        let mut rank_counts = [0 as u8; 13];
        let mut return_data = (0, u8::MAX); // hand type, related data

        for &card in &cards
        {
            let rank = card >> 2;
            rank_counts[rank as usize] += 1;
        }

        let mut pair_counter: u8 = 0;
        let mut trip_counter: u8 = 0;
        let mut quad_counter: u8 = 0; // purely for good looking code, can't get over 1

        let mut pair_rank: u8 = u8::MAX; // for one pair calculations
        let mut trip_rank: u8 = u8::MAX; // for trip vs set calculations

        for rank in 0..13
        {
            if rank_counts[rank] == 2
            {
                pair_counter += 1;
                pair_rank = rank as u8;
            }
            else if rank_counts[rank] == 3
            {
                trip_counter += 1;
                trip_rank = rank as u8;
            }
            else if rank_counts[rank] == 4
            {
                quad_counter += 1;
            }
        }
        
        if pair_counter == 1 && trip_counter == 0 && quad_counter == 0
        {
            return_data.0 = 1; // one pair

            if hand.0 >> 2 != pair_rank && hand.1 >> 2 != pair_rank
            {
                return_data.1 = u8::MAX; // board pair, get out of here
            }
            else if pair_rank > board[0] >> 2
            {
                return_data.1 = 1; // overpair
            }
            else if pair_rank == board[0] >> 2
            {
                return_data.1 = 2; // top pair
            }
            else if pair_rank >= board[1] >> 2
            {
                return_data.1 = 3; // second pair (or second underpair)
            }
            else
            {
                return_data.1 = 4; // weak pair
            }


        }
        else if pair_counter == 2 && trip_counter == 0 && quad_counter == 0
        {
            return_data.0 = 2; // two pair
        }
        else if pair_counter == 0 && trip_counter == 1 && quad_counter == 0
        {
            if hand.0 >> 2 != trip_rank && hand.1 >> 2 != trip_rank
            {
                return_data.0 = 3; // three of a kind
                return_data.1 = 0; // L board trip
            }
            else if hand.0 >> 2 == trip_rank && hand.1 >> 2 == trip_rank
            {
                return_data.0 = 4; // set, yeah we treat them differently

                if trip_rank == board[0] >> 2
                {
                    return_data.1 = 1; // top set
                }
                else if trip_rank == board[1] >> 2
                {
                    return_data.1 = 2; // second set 
                }
                else
                {
                    return_data.1 = 3; // low set
                }
            }
            else
            {
                return_data.0 = 3; // three of a kind
                return_data.1 = 1; // trip
            }
        }
        else if (pair_counter == 1 && trip_counter == 1) || trip_counter == 2
        {
            return_data.0 = 5; // full house
        }
        else if pair_counter == 0 && trip_counter == 0 && quad_counter == 1
        {
            return_data.0 = 6; // four of a kind
        }

        return_data
    }

    fn evaluate_straight(hand: &(Card, Card), board: &Vec<Card>) -> (u8, u8)
    {
        println!("straight eval start");

        let mut cards = vec![hand.0, hand.1];
        cards.extend(board);
        cards.sort_unstable_by(|a, b| a.cmp(&b)); // ascending order because can override stuff for free


        let cards_working = depair(&derank(&cards));
        let board_deranked = derank(&board);
        
        println!("{:?}", cards_working);

        let mut result: (u8, u8) = (0, 0);

        const STRAIGHT_PATTERN: [u8; 4] = [1, 1, 1, 1];
        const WHEEL_BOTTOM: [u8; 4] = [0, 1, 2, 3];

        let mut straight_height: u8 = 0;

        if cards_working.len() < 5
        {
            return result;
        }
        else if cards_working[0..4] == WHEEL_BOTTOM && cards_working[cards_working.len() - 1] == 12
        {
            result.0 = 1;
            straight_height = 3;
        }
        else
        {
            for i in 0..cards_working.len() - 4
            {
                let pattern = [
                    cards_working[i + 1] - cards_working[i], 
                    cards_working[i + 2] - cards_working[i + 1], 
                    cards_working[i + 3] - cards_working[i + 2],
                    cards_working[i + 4] - cards_working[i + 3]
                    ];

                if pattern == STRAIGHT_PATTERN
                {
                    result.0 = 1;
                    straight_height = cards_working[i + 4];
                }
            }
        }

        let mut straight_height_top: u8 = u8::MAX;
        let mut straight_height_second: u8 = u8::MAX;


        for r in -1..9
        {
            if r == -1 // whell 
            {
                let test_arr: [u8; 5] = [0, 1, 2, 3, 12];

                let mut counter: u8 = 0;
                for i in 0..4
                {
                    if board_deranked.contains(&test_arr[i])
                    {
                        counter += 1;
                    }
                }

                if counter == 5
                {
                    if straight_height == 3
                    {
                        straight_height = 0; // BoArD wHeEl Is NoT a SeRiOuS wHeEl
                    }
                }
                else if counter >= 3
                {
                    straight_height_top = 3;
                }
            }
            else
            {
                let r = r as u8;

                let test_arr: [u8; 5] = [r, r + 1, r + 2, r + 3, r + 4];
                let height = r + 4;

                let mut counter: u8 = 0;
                for i in 0..5
                {
                    if board_deranked.contains(&test_arr[i])
                    {
                        counter += 1;
                    }
                }

                println!("{counter}");

                if counter == 5
                {
                    if height >= straight_height
                    {
                        straight_height = 0; // board straight is not a serious straight, straights below board straight are even less serious
                    }

                    straight_height_top = u8::MAX;
                    straight_height_second = u8::MAX;
                    // it didn't affect anything but variable values are now snappy and look nice
                }
                else if counter >= 3
                {
                    straight_height_second = straight_height_top;
                    straight_height_top = height;
                }
            }
        }

        println!("{straight_height_top}");

        // returning shit

        if straight_height == straight_height_top
        {
            result.1 = 1;
        }
        else if straight_height == straight_height_second
        {
            result.1 = 2;
        }
        else
        {
            result.1 = 3;
        }

        result
    }

    fn evaluate_flush (hand: &(Card, Card), board: &Vec<Card>) -> (bool, u8)
    {
        let mut cards = vec![hand.0, hand.1];
        cards.extend(board);

        let mut suit_counts = [0; 4];
        let mut return_data = (false, 0); // is flush, rank of flush
        let mut is_flush = false;
        let mut flush_suit: u8 = u8::MAX;

        for &card in &cards
        {
            let suit = card % 4;
            suit_counts[suit as usize] += 1;

            if suit_counts[suit as usize] >= 5
            {
                is_flush = true;
                flush_suit = suit;

                println!("flush detected");
            }
        }

        if !is_flush
        {
            return return_data;
        }
        else
        {
            return_data.0 = true;
        }

        let mut topcard: u8 = u8::MAX;

        if (hand.0 % 4 == flush_suit) && (hand.1 % 4 == flush_suit)
        {
            if hand.0 > hand.1 
            {
                topcard = hand.0 >> 2;
            }
            else
            {
                topcard = hand.1 >> 2;
            }
        }
        else if hand.0 % 4 == flush_suit
        {
            topcard = hand.0 >> 2;
        }
        else if hand.1 % 4 == flush_suit
        {
            topcard = hand.1 >> 2;
        }
        else
        {
            return (false, 0);
        }

        let mut nuttiness: u8 = 1;
        for checkrank in (0..13).rev()
        {
            let checkcard = checkrank << 2 + flush_suit;

            if board.contains(&checkcard)
            {
                continue;
            }
            
            if topcard == checkrank
            {
                return_data.1 = nuttiness;
                break;
            }

            nuttiness += 1;

            if nuttiness == 3
            {
                break;
            }
        }

        return_data
    }

    fn humble_straight_flush (hand: &(Card, Card), board: &Vec<Card>) -> bool
    {
        let mut cards = vec![hand.0, hand.1];
        cards.extend(board);
        cards.sort_unstable_by(|a, b| b.cmp(&a)); // this is definitely not sorted

        let mut suit_line: u8 = 0;
        let mut line_suit: u8 = u8::MAX;
        let mut last_rank: u8 = u8::MAX;

        for card in cards
        {
            let suit = card % 4;
            let rank = card >> 2;

            if line_suit == suit && last_rank == rank + 1
            {
                suit_line += 1;
                last_rank = rank;

                if suit_line == 5
                {
                    return true;
                }
            }
            else if last_rank == rank
            {
                continue;
            }
            else
            {
                line_suit = suit;
                last_rank = rank;
                suit_line = 0;
            }
        }

        return false;
    }

    println!("{:?}", ret);

    ret
}

/// pocket cards eval
fn evaluate_class (hand: &(Card, Card)) -> u8
{
    let rank1 = hand.0 >> 2;
    let rank2 = hand.1 >> 2;
    let suit1 = hand.0 % 4;
    let suit2 = hand.1 % 4;
    
    if rank1 == rank2
    {
        0
    }
    else if suit1 == suit2
    {
        1
    }
    else
    {
        2
    }
}


/// draw eval
fn evaluate_draw (hand: &(Card, Card), board: &Vec<Card>) -> ([bool; 4], [bool; 4])
{
    let mut cards = vec![hand.0, hand.1];
    cards.extend(board);

    let board_sorted = {
        let mut tmp = board.clone();
        tmp.sort_unstable_by(|a, b| b.cmp(&a));
        tmp
    };

    let mut presence: [bool; 4] = [false, false, false, false]; // overcards, gutshot, open ended, flush draw present
    let mut bothhole: [bool; 4] = [false, false, false, false]; // same but for if both hole cards are used

    let rank = evaluate_rank(hand, &board_sorted);

    let over_data = eval_overcards(hand, &board_sorted);
    let gtsh_data = eval_gutshot(hand, &board_sorted);
    let oesd_data = eval_oesd(hand, &board_sorted);
    let fdrw_data = eval_fdraw(hand, &board_sorted);

    if rank.0 < 3 // two pair or less
    {
        presence[0] = over_data.0;
        bothhole[0] = over_data.1;
    }
    if rank.0 < 5 // no straight
    {
        if oesd_data.0
        {
            presence[2] = oesd_data.0;
            bothhole[2] = oesd_data.1;
        }
        else
        {
            presence[1] = gtsh_data.0;
            bothhole[1] = gtsh_data.1;
        }
    }
    if rank.0 < 6 // no flush
    {
        presence[3] = fdrw_data.0;
        bothhole[3] = fdrw_data.1;
    }

    // FUNCS
    
    fn eval_overcards (hand: &(Card, Card), board: &Vec<Card>) -> (bool, bool)
    {
        let mut ret: (bool, bool) = (false, false); // overcard presence, both hole cards are overcards

        if hand.0 >> 2 > board[0] >> 2 || hand.1 >> 2 > board[0] >> 2
        {
            ret.0 = true;
        }
        if hand.0 >> 2 > board[0] >> 2 && hand.1 >> 2 > board[0] >> 2
        {
            ret.1 = true;
        }

        ret
    }

    fn eval_gutshot (hand: &(Card, Card), board: &Vec<Card>) -> (bool, bool)
    {
        let mut compboard = board.clone();
        let mut ret = (false, false);

        compboard.push(hand.0);
        compboard.push(hand.1);

        compboard.sort_unstable_by(|a, b| b.cmp(&a));

        let compboard_deranked = derank(&compboard);

        const GMASK_1: [u8; 3] = [2, 1, 1];
        const GMASK_2: [u8; 3] = [1, 2, 1];
        const GMASK_3: [u8; 3] = [1, 1, 2];

        for i in 0..(compboard.len()-3)
        {
            let mask = [
                compboard_deranked[i+1]-compboard_deranked[i],
                compboard_deranked[i+2]-compboard_deranked[i+1],
                compboard_deranked[i+3]-compboard_deranked[i+2]
            ];

            if mask == GMASK_1 || mask == GMASK_2 || mask == GMASK_3
            {
                let mut isone = false;

                if
                    compboard[i] == hand.0 ||
                    compboard[i+1] == hand.0 ||
                    compboard[i+2] == hand.0 ||
                    compboard[i+3] == hand.0
                {
                    ret.0 = true;
                    isone = true;
                }

                if
                    compboard[i] == hand.1 ||
                    compboard[i+1] == hand.1 ||
                    compboard[i+2] == hand.1 ||
                    compboard[i+3] == hand.1
                {
                    if isone
                    {
                        ret.1 = true;
                    }
                    else
                    {
                        ret.0 = true;
                    }
                }
            }
        }
        
        ret
    }


    fn eval_oesd (hand: &(Card, Card), board: &Vec<Card>) -> (bool, bool)
    {
        let mut compboard = board.clone();
        let mut ret = (false, false);

        compboard.push(hand.0);
        compboard.push(hand.1);

        compboard.sort_unstable_by(|a, b| b.cmp(&a));

        let compboard_deranked = derank(&compboard);

        const OEMASK: [u8; 3] = [1, 1, 1];
        const DGMASK: [u8; 4] = [2, 1, 1, 2];

        // normal openender
        for i in 0..(compboard.len()-3)
        {
            if compboard_deranked[i] == 0 && compboard_deranked[i+3] == 12
            {
                continue; // no self-respecting oesd has twos or aces
            }

            let mask = [
                compboard_deranked[i+1]-compboard_deranked[i],
                compboard_deranked[i+2]-compboard_deranked[i+1],
                compboard_deranked[i+3]-compboard_deranked[i+2]
            ];

            if mask == OEMASK
            {
                let mut isone = false;

                if
                    compboard[i] == hand.0 ||
                    compboard[i+1] == hand.0 ||
                    compboard[i+2] == hand.0 ||
                    compboard[i+3] == hand.0
                {
                    ret.0 = true;
                    isone = true;
                }

                if
                    compboard[i] == hand.1 ||
                    compboard[i+1] == hand.1 ||
                    compboard[i+2] == hand.1 ||
                    compboard[i+3] == hand.1
                {
                    if isone
                    {
                        ret.1 = true;
                    }
                    else
                    {
                        ret.0 = true;
                    }
                }
            }
        }
        
        // dgsd
        for i in 0..(compboard.len()-4)
        {
            let mask = [
                compboard_deranked[i+1]-compboard_deranked[i],
                compboard_deranked[i+2]-compboard_deranked[i+1],
                compboard_deranked[i+3]-compboard_deranked[i+2],
                compboard_deranked[i+4]-compboard_deranked[i+3]
            ];

            if mask == DGMASK
            {
                let mut isone = false;

                if
                    compboard[i] == hand.0 ||
                    compboard[i+1] == hand.0 ||
                    compboard[i+2] == hand.0 ||
                    compboard[i+3] == hand.0 ||
                    compboard[i+4] == hand.0
                {
                    ret.0 = true;
                    isone = true;
                }

                if
                    compboard[i] == hand.1 ||
                    compboard[i+1] == hand.1 ||
                    compboard[i+2] == hand.1 ||
                    compboard[i+3] == hand.1 ||
                    compboard[i+4] == hand.1
                {
                    if isone
                    {
                        ret.1 = true;
                    }
                    else
                    {
                        ret.0 = true;
                    }
                }
            }
        }
        
        ret
    }


    fn eval_fdraw (hand: &(Card, Card), board: &Vec<Card>) -> (bool, bool)
    {
        let mut suits = [0, 0, 0, 0];

        for card in board
        {
            suits[(card % 4) as usize] += 1;
        }

        for i in 0..4
        {
            if suits[i] == 3 && hand.0 % 4 == i as u8 && hand.1 % 4 == i as u8
            {
                return (true, true);
            }
            else if suits[i] == 4 && (hand.0 % 4 == i as u8 || hand.1 % 4 == i as u8)
            {
                return (true, false);
            }

        }

        (false, false)
    }

    (presence, bothhole)
}

/// Removes the suit information from a card vector, leaving only the ranks.
fn derank(cards: &Vec<u8>) -> Vec<u8>
{
    let mut deranked: Vec<u8> = vec![];

    for card in cards
    {
        deranked.push(card >> 2);
    }

    deranked
}

/// Removes paires from deranked and sorted card vec
fn depair(cards: &Vec<u8>) -> Vec<u8>
{
    let mut depaired: Vec<u8> = vec![];

    let mut last_card = u8::MAX;

    for card in cards
    {
        if *card != last_card
        {
            depaired.push(*card);
            last_card = *card;
        }
    }

    depaired
}
