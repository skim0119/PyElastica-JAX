# PyElastica-JAX

Domain language for Cosserat-rod simulation on JAX, including how simulation state is exposed and persisted.

## Language

### Simulation objects

**Rod**:
A Cosserat continuum object the user creates and holds as a host-side representation (`CosseratRod`). Finalization packs its state into a Block; afterward its arrays are detached from and are not authoritative for the running simulation. A Rod can be an explicit synchronization source or target.
_Avoid_: rod view (when meaning the user-held CosseratRod), reference rod as a separate type

**Block**:
The packed memory container that owns authoritative runtime state, memory layout, and device placement for one or more rods during integration.
_Avoid_: memory block as a distinct concept from Block; shard (except MPI-local wrappers)

**Rod-local state**:
A transient, rod-shaped projection of authoritative Block state used during Rod-local computation. It is not the user-held Rod and does not transfer state to host memory.
_Avoid_: Rod (the host-side object); rod view when referring to the user-held Rod

**Simulator**:
The system collection that registers rods, enables block supports, and after finalize iterates blocks for stepping.
_Avoid_: environment (except in RL/example wrappers)

### Persistence

**Save**:
The operation that opens one HDF5 file, writes file-level attributes, and delegates payload groups to an `Hdf5StateIO` implementation for the target—without calling `from_device` / `to_device`. Source of truth follows the target: a **Block** (including blocks reached via a **Simulator** with the `Hdf5IO` mixin) is saved from device state, with device metadata; a user-held **Rod** is adapted and saved from its host arrays as-is (may be unsynced with any simulator). For a block, the written layout includes ghost segments so a matching Load can restore packed state exactly.
_Avoid_: dump, export (prefer Save); auto-sync before Save; stripping ghosts for a “clean” viz layout inside Save; treating a Rod Save as a simulator resume artifact; type-switching on rod/block/simulator inside Save

**Load**:
The inverse of Save. Loading a block/simulator file can resume integration when the file’s schema level is 10; by default the file’s JAX platform and device id must match the current runtime, with an option to disable that check. Loading a rod file only refreshes that rod’s host arrays — it does not rehydrate simulator/block state, even if the rod file’s schema level is 10.
_Avoid_: restore (except colloquially for restart workflows); loading a non-10 block file as a restart; assuming a rod Load syncs the simulator

**Device metadata**:
JAX platform and device id recorded on a block/simulator Save so Load can verify the runtime before restoring device state.
_Avoid_: storing only platform; treating device metadata as required on rod Saves

**Schema level**:
An integer recorded on the file that selects which field groups are included in a Save. Higher levels add fields; they do not replace the lower set. Level 10 on a **Rod** means all host collection fields for that rod; it does not make the file a simulator resume artifact.

| Level | Collection names |
| ----- | ---------------- |
| 0 | `position_collection`, `director_collection`, `radius` |
| 1 | level 0 + `sigma`, `kappa`, `velocity_collection`, `omega_collection` |
| 10 | full state for that target; block/simulator files with level 10 are Load-safe for resume |

_Avoid_: verbose / verbosity as the domain name (implementation kwarg may use `verbose`); log level; treating a rod level-10 file as simulator restart; treating block levels other than 10 as restartable; adding extra viz fields to 0/1 beyond this table

**Collection name**:
The canonical field identifier on rods and blocks (`position_collection`, `director_collection`, `omega_collection`, …). Schema levels speak in collection names, not physics shorthand.
_Avoid_: angular_velocity (use `omega_collection`); director alone (use `director_collection`); position alone in schema docs (use `position_collection`)
